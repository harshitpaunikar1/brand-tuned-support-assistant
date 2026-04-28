"""
FastAPI service for the brand-tuned support assistant.
Serves the LoRA-adapted model or falls back to the base model for response generation.
"""
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    max_new_tokens: int = 150


class ChatResponse(BaseModel):
    reply: str
    model_type: str
    latency_ms: float


class ModelServer:
    """Loads and serves a LoRA-tuned or base model for inference."""

    def __init__(self, model_path: str, model_type: str = "lora_tuned"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """Load tokenizer and model from disk."""
        if not PEFT_AVAILABLE:
            logger.warning("transformers/peft not installed. ModelServer cannot load.")
            return False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model_type == "lora_tuned" and Path(self.model_path).exists():
                base_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                base_model = AutoModelForCausalLM.from_pretrained(base_name, low_cpu_mem_usage=True)
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True)
            self.model.eval()
            self._loaded = True
            logger.info("Model loaded from %s (type: %s)", self.model_path, self.model_type)
            return True
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate a response for the given prompt."""
        if not self._loaded:
            return "Model not loaded."
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip the input prompt from the output
        if prompt in full_text:
            return full_text[len(prompt):].strip()
        return full_text.strip()


if FASTAPI_AVAILABLE:
    app = FastAPI(title="Brand-Tuned Support Assistant", version="1.0")
    server = ModelServer(model_path="brand_adapter_demo", model_type="lora_tuned")

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if not server._loaded:
            raise HTTPException(status_code=503, detail="Model not loaded. Call POST /load first.")
        prompt = f"### Instruction:\n{request.message}\n\n### Response:\n"
        t0 = time.perf_counter()
        reply = server.generate(prompt, max_new_tokens=request.max_new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000
        return ChatResponse(reply=reply, model_type=server.model_type, latency_ms=round(latency_ms, 1))

    @app.post("/load")
    async def load_model():
        success = server.load()
        return {"loaded": success, "model_type": server.model_type}

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": server._loaded, "model_type": server.model_type}

    @app.get("/model-info")
    async def model_info():
        return {"model_path": server.model_path, "type": server.model_type, "loaded": server._loaded}

else:
    app = None


if __name__ == "__main__":
    # Run with: uvicorn serve:app --host 0.0.0.0 --port 8000
    print("Start server with: uvicorn serve:app --host 0.0.0.0 --port 8000")
    print("Then POST /load to initialize the model before calling /chat")
