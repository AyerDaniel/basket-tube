from huggingface_hub import login                                                                                                                                                           
login(token="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")  

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
DATASET_PATH   = Path("my_basket_tube/csv/training_dataset.json")
MODEL_DIR      = Path("my_basket_tube/models/paligemma_lora")
BASE_MODEL     = "google/paligemma-3b-pt-224"
EPOCHS         = 3
BATCH_SIZE     = 1    # small — VLM is large
LR             = 2e-4
MAX_NEW_TOKENS = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA targets for PaliGemma (language model attention layers)
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


# =============================================================================
# DATASET
# =============================================================================

def concat_frames_horizontal(frame_paths):
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    total_w = sum(img.width for img in imgs)
    max_h   = max(img.height for img in imgs)
    canvas  = Image.new("RGB", (total_w, max_h))
    x = 0
    for img in imgs:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


class BasketTubeDataset(Dataset):
    def __init__(self, data, processor):
        self.entries   = list(data.values())
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        image = concat_frames_horizontal(entry['frame_paths'])

        prompt = (
            f"Commentary segment at [{entry['timestamp']}]: \"{entry['subtitle']}\"\n\n"
            "Please analyze this moment."
        )
        target = entry['response']

        # PaliGemma processor handles image + text jointly
        encoding = self.processor(
            images=image,
            text=prompt,
            suffix=target,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}


# =============================================================================
# TRAINING
# =============================================================================

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first.")
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    print(f"Loaded dataset: {len(data)} entries")
    return data


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()

    # 4-bit quantization — keeps PaliGemma-3B within 16GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading {BASE_MODEL}...")
    processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config
    )

    # LoRA on language model attention
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Resume from checkpoint if available
    adapter_path = MODEL_DIR / "adapter_model"
    if adapter_path.exists():
        model.load_adapter(str(adapter_path))
        print("Resumed from checkpoint.")

    dataset    = BasketTubeDataset(data, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=len(dataloader) * EPOCHS
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss    = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

        # Save after each epoch
        model.save_pretrained(str(adapter_path))
        processor.save_pretrained(str(MODEL_DIR / "processor"))
        print(f"Saved checkpoint → {adapter_path}")

    print(f"\nFine-tuning complete. Adapter saved to {adapter_path}")


# =============================================================================
# INFERENCE
# =============================================================================

def load_model_for_inference():
    adapter_path = MODEL_DIR / "adapter_model"
    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_path}. Run train() first.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    processor = PaliGemmaProcessor.from_pretrained(str(MODEL_DIR / "processor"))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config
    )
    model.load_adapter(str(adapter_path))
    model.eval()

    return model, processor


def analyze_moment(frame_paths, subtitle, timestamp, model, processor):
    image  = concat_frames_horizontal(frame_paths)
    prompt = (
        f"Commentary segment at [{timestamp}]: \"{subtitle}\"\n\n"
        "Please analyze this moment."
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    # Decode only the generated portion
    input_len = inputs['input_ids'].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    train()
