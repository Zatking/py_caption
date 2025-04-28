import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def prepare_dataset(image_dir, caption_json, output_json):
    with open(caption_json, "r", encoding="utf-8") as f:
        all_captions = json.load(f)

    if not all_captions:
        print("⚠️ File caption trống hoặc không hợp lệ!")
        return

    dataset = []
    for disease_label in os.listdir(image_dir):
        disease_dir = os.path.join(image_dir, disease_label)

        if os.path.isdir(disease_dir):
            image_files = [f for f in os.listdir(disease_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_name in image_files:
                img_path = os.path.join(disease_dir, img_name)
                caption_key = os.path.join(disease_label, img_name)

                caption = all_captions.get(caption_key)
                if caption:
                    dataset.append({
                        "image": img_path,
                        "caption": caption,
                        "label": disease_label 
                    })
                else:
                    print(f"⚠️ Cảnh báo: Không tìm thấy caption cho {caption_key}")

    if not dataset:
        print("⚠️ Không có dataset hợp lệ để lưu!")
        return

    print(f"Mẫu dữ liệu: {dataset[:2]}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Dataset đã được lưu vào {output_json}")

if __name__ == "__main__":
    prepare_dataset("app/static/images/test_set", "app/static/caption.json", "app/static/prepared_dataset.json")
