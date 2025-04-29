import os
import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW

# =======================
# STEP 1: Prepare Dataset
# =======================
def prepare_dataset(image_dir, caption_json, output_json):
    with open(caption_json, "r", encoding="utf-8") as f:
        all_captions = json.load(f)

    if not all_captions:
        print("⚠️ File caption trống hoặc không hợp lệ!")
        return []

    dataset = []
    image_dir = Path(image_dir)

    for image_path in image_dir.rglob("*.[jp][pn]g"):  # jpg, png
        rel_path = image_path.relative_to(image_dir).as_posix()
        disease_label = image_path.parent.name

        caption = all_captions.get(rel_path)
        if caption:
            dataset.append({
                "image": str(image_path),
                "caption": caption[0] if isinstance(caption, list) else caption,
                "label": disease_label
            })
        else:
            print(f"⚠️ Không tìm thấy caption cho {rel_path}")

    if not dataset:
        print("⚠️ Không có dữ liệu hợp lệ để huấn luyện!")
        return []

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu dataset vào {output_json} với {len(dataset)} mẫu.")
    return dataset

# =======================
# STEP 2: Dataset & Preprocess
# =======================
class CaptionDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        inputs = self.processor(images=image, text=item["caption"], return_tensors="pt", padding="max_length", truncation=True)
        image_inputs = self.processor(images=image, return_tensors="pt")
        text_inputs = self.processor(text=item["caption"], return_tensors="pt", padding="max_length", truncation=True)

        return {
        "pixel_values": image_inputs["pixel_values"].squeeze(0),
        "input_ids": text_inputs["input_ids"].squeeze(0),
        "attention_mask": text_inputs["attention_mask"].squeeze(0),
        "labels": text_inputs["input_ids"].squeeze(0)  # dùng chính input làm label
        }

# =======================
# STEP 3: Training
# =======================
def train(model, processor, dataset, epochs=5, batch_size=4, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chuyển mô hình lên GPU    
    model.to(device)

    dataloader = DataLoader(CaptionDataset(dataset, processor), batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr,weight_decay=0.01, eps=1e-8)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Chuyển batch dữ liệu lên GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                        
                            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"📚 Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

# =======================
# STEP 4: Inference
# =======================
def predict(image_path, model, processor):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chuyển mô hình lên GPU
    model.to(device)
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)  # Chuyển input lên GPU
    
    out = model.generate(**inputs)
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

    # Bước 1: Chuẩn bị dữ liệu
    dataset = prepare_dataset(image_dir, caption_json, output_json)
    if not dataset:
        exit()

    # Bước 2: Load BLIP model + processor
    print("🚀 Đang load mô hình BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Bước 3: Fine-tune nhanh
    print("🧠 Bắt đầu fine-tune mô hình...")
    train(model, processor, dataset, epochs=4)  # bạn có thể tăng số epoch

    # Bước 4: Dự đoán thử caption ảnh đầu tiên
    print("\n🔍 Dự đoán caption ảnh đầu tiên:")
    predict(dataset[0]["image"], model, processor)
