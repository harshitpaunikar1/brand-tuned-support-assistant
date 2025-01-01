# Brand-Tuned Support Assistant

This repository documents a lightweight fine-tuning workflow for turning messy support conversations into a customer-support assistant that answers in the company's tone.

## Domain
E-commerce / Customer Support

## Overview
Balanced local small-model adaptation with a hosted baseline so the team could compare control, cost, and response quality honestly.

## Methodology
1. Audited support chats, FAQs, ticket labels, and tone inconsistencies to define what the assistant should learn and what data needed masking.
2. Built data-preparation flow to remove duplicates, strip sensitive content, and convert cleaned conversations into instruction-response examples.
3. Selected a small-model path around TinyLlama and Qwen-class models with LoRA-style tuning to stay within limited hardware constraints.
4. Used Transformers, Datasets, and PEFT to fine-tune adapters, validate outputs, and compare behaviour against Gemini Flash as a baseline.
5. Focused evaluation on correctness, tone consistency, and domain fit so the client could decide whether local tuning was worth the effort.
6. Wrapped the tuned model path in a simple FastAPI service to make downstream integration realistic for future product use.

## Skills
- LLM Fine-Tuning
- Transformers
- PEFT / LoRA
- TinyLlama / Qwen
- Instruction Dataset Design
- Data Cleaning
- Gemini Flash Benchmarking
- FastAPI

## Source
This README was generated from the portfolio project data used by `/Users/harshitpanikar/Documents/Test_Projs/harshitpaunikar1.github.io/index.html`.
