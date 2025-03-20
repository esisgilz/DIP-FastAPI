import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import io
import os

# Define storage directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

# Define DIP Model (Replace with ResNet if needed)
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Load model
model = DIPModel()
model.eval()

# FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "DIP FastAPI Running!"}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    output_img = transforms.ToPILImage()(output.squeeze(0))

    # Save processed image in uploads directory
    output_filename = f"{UPLOAD_DIR}/denoised_{file.filename}"
    output_img.save(output_filename)

    return {
        "filename": file.filename,
        "output": output_filename,
        "download_url": f"/download/{os.path.basename(output_filename)}"
    }

@app.get("/download/{filename}")
async def download_image(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "File not found"}
