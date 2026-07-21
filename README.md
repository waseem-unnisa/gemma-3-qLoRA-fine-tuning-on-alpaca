# 🔧 Gemma 3 QLoRA Fine-Tuning on Alpaca

Fine-tuning Google's **Gemma 3 (1B)** model on the **Alpaca instruction-following dataset** using **QLoRA** — trained entirely on a free Google Colab T4 GPU.

This project demonstrates an end-to-end LLM fine-tuning workflow: loading a quantized base model, attaching LoRA adapters, formatting a dataset into a chat template, training, evaluating, and publishing the result to the Hugging Face Hub.

---

## 🧠 What This Project Covers

- **QLoRA fine-tuning** — combining 4-bit quantization with Low-Rank Adaptation to fine-tune a language model on consumer-grade/free GPU hardware
- **Parameter-efficient training** — only ~31M trainable parameters out of ~5.1B (0.6%), keeping training fast and memory-light
- **Dataset formatting** — converting raw instruction/input/output data into a model-specific chat template
- **Practical debugging** — this project involved working through real constraints of free-tier GPU training: VRAM limits, dtype mismatches, and model-hardware compatibility issues (documented in the notebook)

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Base model | [Gemma 3 (1B, instruction-tuned)](https://huggingface.co/google/gemma-3-1b-it) |
| Fine-tuning framework | [Unsloth](https://github.com/unslothai/unsloth) |
| Training loop | [TRL (SFTTrainer)](https://github.com/huggingface/trl) |
| Dataset | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (52K instruction-following examples) |
| Hardware | Google Colab — free T4 GPU |
| Adapter format | LoRA (via PEFT) |

---

## 📊 Training Configuration

| Setting | Value |
|---|---|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max sequence length | 1024 |
| Batch size (per device) | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| Training steps | 60 |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |

> Note: training was limited to 60 steps (a subset of the full 52K-example dataset) to fit within a short, free-tier Colab session. Loss dropped steadily from ~2.3 to ~1.6 over the run — enough to demonstrate the fine-tuning pipeline works correctly, though a longer run over more epochs would improve output quality further.

---

## 🚀 Try It Yourself

The trained LoRA adapter is published on Hugging Face:

**[wazym/gemma3-alpaca-lora](https://huggingface.co/wazym/gemma3-alpaca-lora)**

python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "wazym/gemma3-alpaca-lora",
    max_seq_length = 1024,
    load_in_4bit = True,
)

FastModel.for_inference(model)

messages = [{"role": "user", "content": "What are the three primary colors?"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))




## 📓 Notebook
The full training process ... is in [`Fine_Tune_Gemma_3_1B.ipynb`](./Fine_Tune_Gemma_3_1B.ipynb).

To run it yourself: open in Google Colab, set the runtime to **T4 GPU**, and run all cells top to bottom.

---

## 🗺️ What's Next

- [ ] Train for more steps / full epochs for improved output quality
- [ ] Evaluate on a held-out test set with quantitative metrics
- [ ] Try a domain-specific dataset instead of general instruction-following
- [ ] Merge adapter into the base model and export to GGUF for local inference

---

##  Author

- **GitHub:** [waseem-unnisa](https://github.com/waseem-unnisa)
- **LinkedIn:** [Waseem Unnisa](https://www.linkedin.com/in/waseem-unnisa-8a68293ba/)
- **Hugging Face:** [wazym](https://huggingface.co/wazym)