from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import threading
app = FastAPI()
from app.test import process_images



@app.get("/")
async def root():
    return {"message": "Image Processing API is running!"}

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=process_images)
    thread.start()
