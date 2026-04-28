"""
LoRA fine-tuning pipeline for brand-tuned customer support assistant.
Uses PEFT adapters on TinyLlama or Qwen-class models.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("transformers / peft not installed. Fine-tuning will not run.")


@dataclass
class FinetuneConfig:
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "brand_tuned_adapter"
    max_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    eval_ratio: float = 0.1


class DatasetBuilder:
    """
    Prepares instruction-response training data from raw support conversations.
    Handles PII masking, deduplication, and format conversion.
    """

    PII_PATTERNS = [
        (re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"), "[EMAIL]"),
        (re.compile(r"\b\d{10,}\b"), "[PHONE]"),
        (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD]"),
    ]

    def __init__(self, raw_conversations: List[Dict]):
        self.raw_conversations = raw_conversations

    def clean_conversation(self, conv: Dict) -> Dict:
        """Strip PII and normalize whitespace."""
        cleaned = {}
        for k, v in conv.items():
            if isinstance(v, str):
                for pattern, replacement in self.PII_PATTERNS:
                    v = pattern.sub(replacement, v)
                v = " ".join(v.split())
            cleaned[k] = v
        return cleaned

    def convert_to_instruction_format(self, conv: Dict) -> Dict:
        """Convert a user/assistant conversation pair to instruction-response format."""
        return {
            "instruction": conv.get("user", conv.get("question", "")),
            "input": "",
            "output": conv.get("assistant", conv.get("answer", "")),
        }

    def build_dataset(self) -> Optional[object]:
        """Clean, deduplicate, and convert conversations into a HuggingFace Dataset."""
        cleaned = [self.clean_conversation(c) for c in self.raw_conversations]
        seen = set()
        unique = []
        for c in cleaned:
            key = (c.get("user", ""), c.get("assistant", ""))
            if key not in seen and c.get("user") and c.get("assistant"):
                seen.add(key)
                unique.append(c)
        formatted = [self.convert_to_instruction_format(c) for c in unique]
        if not PEFT_AVAILABLE:
            logger.warning("Dataset built in-memory but HuggingFace Dataset not available.")
            return formatted
        return Dataset.from_list(formatted)

    def split_train_eval(self, dataset, eval_ratio: float = 0.1):
        """Split dataset into train and eval portions."""
        if isinstance(dataset, list):
            n = len(dataset)
            split = max(1, int(n * eval_ratio))
            return dataset[split:], dataset[:split]
        split = dataset.train_test_split(test_size=eval_ratio, seed=42)
        return split["train"], split["test"]


class LoRAFineTuner:
    """
    Fine-tunes a small LLM with LoRA adapters using the PEFT library.
    Compares tuned model outputs against a baseline for evaluation.
    """

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_base_model(self):
        """Load tokenizer and base model, then wrap with LoRA configuration."""
        if not PEFT_AVAILABLE:
            raise RuntimeError("Install transformers and peft to run fine-tuning.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name, low_cpu_mem_usage=True
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        return self.model

    def prepare_training_args(self) -> object:
        """Build HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            report_to="none",
        )

    def tokenize_function(self, examples: Dict) -> Dict:
        prompts = [
            f"### Instruction:\n{i}\n\n### Response:\n{o}"
            for i, o in zip(examples["instruction"], examples["output"])
        ]
        return self.tokenizer(
            prompts, truncation=True, max_length=self.config.max_length, padding="max_length"
        )

    def train(self, train_dataset, eval_dataset) -> str:
        """Fine-tune the model and save the adapter to disk."""
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(self.tokenize_function, batched=True)
        args = self.prepare_training_args()
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
        )
        trainer.train()
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        return self.config.output_dir

    def compare_outputs(self, prompts: List[str], baseline_fn: Callable) -> List[Dict]:
        """Generate from tuned model and compare against a baseline callable."""
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output_ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
            tuned_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            baseline_response = baseline_fn(prompt)
            results.append({
                "prompt": prompt,
                "tuned": tuned_response,
                "baseline": baseline_response,
            })
        return results


if __name__ == "__main__":
    sample_convos = [
        {"user": "Where is my order?", "assistant": "Your order is being processed. Check your email for tracking details."},
        {"user": "I want a refund.", "assistant": "We can process your refund within 3-5 business days. Please share your order number."},
    ]
    config = FinetuneConfig(output_dir="brand_adapter_demo")
    builder = DatasetBuilder(sample_convos)
    dataset = builder.build_dataset()
    print(f"Dataset built with {len(dataset) if dataset else 0} examples.")
    print("To run fine-tuning, install transformers and peft, then call LoRAFineTuner(config).load_base_model() and .train()")
