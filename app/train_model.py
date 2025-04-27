import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import push_to_hub
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Đường dẫn thư mục ảnh và tệp JSON chứa caption
IMAGE_DIR = "app/static/images/test_set"
OUTPUT_JSON = "app/static/caption.json"

# Thư mục chứa ảnh kiểm tra mới
TEST_IMAGE_PATH = "app/static/images/test/FU-athlete-foot (7).jpg" 

class CaptionDataset(Dataset):
    def __init__(self, image_dir, caption_json, transform=None):
        with open(caption_json, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = os.path.join(self.image_dir, item['image_file'])
        image = Image.open(image_file).convert("RGB")
        caption = item['caption']
        if self.transform:
            image = self.transform(image)
        return image, caption

def prepare_dataloader(image_dir, output_json):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CaptionDataset(image_dir, output_json, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader

def load_pretrained_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model

def fine_tune_model(image_dir, output_json):
    processor, model = load_pretrained_model()

    dataloader = prepare_dataloader(image_dir, output_json)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        task_type=TaskType.SEQUENCE_CLASSIFICATION,
    )

    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Thêm in chi tiết trong quá trình huấn luyện
    for epoch in range(3):
        print(f"\nStarting epoch {epoch + 1}...")
        epoch_loss = 0.0

        for batch_idx, (images, captions) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch")):
            optimizer.zero_grad()

            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True)
            labels = inputs.input_ids

            outputs = model(input_ids=inputs.input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # In mất mát của mỗi batch
            print(f"Batch {batch_idx + 1}: Loss = {loss.item()}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

    print("Training completed.")
    model.save_pretrained("app/static/fine_tuned_model")
    processor.save_pretrained("app/static/fine_tuned_model")

def save_model_to_hub():
    push_to_hub(repo_id="your_huggingface_repo_id", model_path="app/static/fine_tuned_model")

def generate_caption(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(input_ids=inputs['input_ids'])
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def test_new_image(image_path):
    processor, model = load_pretrained_model()
    caption = generate_caption(image_path, processor, model)
    print(f"Generated caption for the image: {caption}")

def train_and_save_model(image_dir, output_json):
    fine_tune_model(image_dir, output_json)
    # save_model_to_hub()

if __name__ == "__main__":
    # Huấn luyện và lưu mô hình
    train_and_save_model(IMAGE_DIR, OUTPUT_JSON)

    # Kiểm tra ảnh từ thư mục khác
    test_new_image(TEST_IMAGE_PATH)
