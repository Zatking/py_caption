from pathlib import Path
import os
import json
from PIL import Image
import google.generativeai as genai

IMAGE_DIR = "app/static/images/test_set"
OUTPUT_JSON = "app/static/caption.json"
GEMINI_API_KEY = "AIzaSyC3fsF-OPXGmtpp0eUtIo5CVrQeNOlOx7g"
genai.configure(api_key=GEMINI_API_KEY)

def get_all_images(directory):
    return list(Path(directory).rglob("*.jpg")) + \
           list(Path(directory).rglob("*.jpeg")) + \
           list(Path(directory).rglob("*.png"))

def generate_caption_gemini(image_path):
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = """
        Mô tả bức ảnh này bằng tiếng việt, đây là ảnh y khoa nên hãy mô tả thật kỹ.
        Mô tả vị trí, kích thước, màu sắc, kết cấu, độ rõ nét của các cạnh, tính đối xứng, phân bố và sự hiện diện của bất kỳ đặc điểm bất thường nào (ví dụ: vảy, viêm, chảy máu hoặc nhiều màu sắc).
        Tập trung vào ngoại hình lâm sàng giúp phân biệt nó với các tình trạng tương tự khác."""
        response = model.generate_content([prompt, img])

        caption = response.text.replace("\\n", " ").replace("\n", " ")
        caption = " ".join(caption.split())  
        return caption
    except Exception as e:
        print(f"Lỗi khi tạo caption với Gemini: {e}")
        return None

def update_json_file(output_json, image_path, caption):
    """Cập nhật hoặc tạo mới file JSON với caption của ảnh."""
    try:
        caption_data = {str(image_path.relative_to(IMAGE_DIR)): caption}
        if Path(output_json).exists():
            with open(output_json, "r+", encoding="utf-8") as f:
                data = json.load(f)  
                data.update(caption_data)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(caption_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi khi cập nhật file JSON: {e}")

def process_images(output_json=OUTPUT_JSON):
    """Duyệt qua tất cả ảnh, tạo caption và lưu vào file JSON từng ảnh một."""
    all_images = get_all_images(IMAGE_DIR)

    for image_path in all_images:
        try:
            caption = generate_caption_gemini(image_path)
            if caption:
                update_json_file(output_json, image_path, caption)
                print(f"✔️ Đã xử lý: {image_path.relative_to(IMAGE_DIR)}")
            else:
                print(f"Không thể tạo caption cho {image_path}")
        except Exception as e:
            print(f"Không thể xử lý {image_path}: {e}")

    print(f"✅ Tất cả các caption đã được cập nhật trong {output_json}")

if __name__ == "__main__":
    process_images()
