from fastapi import FastAPI
import threading
from app.handle_caption import process_images
from app.train_model import train_and_save_model

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Image Processing API is running!"}

@app.on_event("startup")
def startup_event():
    # Thread cho việc xử lý ảnh
    # thread1 = threading.Thread(target=process_images)
    # thread1.start()

    # Thread cho việc huấn luyện và lưu mô hình
    thread2 = threading.Thread(target=train_and_save_model, args=("app/static/images/test_set", "app/static/caption.json"))
    thread2.start()
