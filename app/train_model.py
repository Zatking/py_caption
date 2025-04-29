import os
import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm

# =======================
# STEP 1: Prepare Dataset
# =======================
def prepare_dataset(image_dir, caption_json, output_json):
    with open(caption_json, "r", encoding="utf-8") as f:
        all_captions = json.load(f)

    dataset = []
    image_dir = Path(image_dir)

    for image_path in image_dir.rglob("*.[jp][pn]g"):
        rel_path = image_path.relative_to(image_dir).as_posix()
        disease_label = image_path.parent.name

        caption = all_captions.get(rel_path)
        if caption:
            dataset.append({
                "image": str(image_path),
                "caption": caption[0] if isinstance(caption, list) else caption,
                "label": disease_label
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu dataset vào {output_json} với {len(dataset)} mẫu.")
    return dataset

# =======================
# STEP 2: Dataset Class
# =======================
class CaptionDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        image = self.transform(image)

        inputs = self.processor(images=image, text=item["caption"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# =======================
# STEP 3: Training
# =======================
def train(model, processor, dataset, epochs=5, batch_size=8, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(CaptionDataset(dataset, processor), batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"📚 Epoch {epoch+1} average loss: {avg_loss:.4f}")

# =======================
# STEP 4: Inference
# =======================
def predict(image_path, model, processor):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"🖼️ Image: {image_path}")
    print(f"📝 Predicted Caption: {caption}")

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    image_dir = "app/static/images/test_set"
    caption_json = "app/static/caption.json"
    output_json = "app/static/prepared_dataset.json"

    dataset = prepare_dataset(image_dir, caption_json, output_json)
    if not dataset:
        exit()

    print("🚀 Đang load mô hình BLIP (Large)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    print("🧠 Bắt đầu fine-tune mô hình...")
    train(model, processor, dataset, epochs=5)

    print("\n🔍 Caption ảnh đầu tiên:")
    predict(dataset[0]["image"], model, processor)
