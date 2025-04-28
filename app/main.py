from fastapi import FastAPI
import threading
from app.handle_caption import process_images
from app.train_model import prepare_dataset

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Image Processing API is running!"}

@app.on_event("startup")
def startup_event():
    """
    Khởi động API và chạy các tác vụ xử lý ảnh và huấn luyện mô hình trong các thread riêng biệt.
    """
    # Thread cho việc xử lý ảnh
    # thread1 = threading.Thread(target=process_images)
    # thread1.start()

    # Thread cho việc chuẩn bị dataset và huấn luyện mô hình
    thread2 = threading.Thread(target=prepare_dataset, args=("app/static/images/test_set", "app/static/caption.json", "app/static/prepared_dataset.json"))
    thread2.start()

    # # Thread cho huấn luyện mô hình
    # thread3 = threading.Thread(target=train_model)
    # thread3.start()

    print("Startup process has started in separate threads.")
