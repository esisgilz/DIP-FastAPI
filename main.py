from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import io
import os

app = FastAPI()

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a modified ResNet-based denoiser
class ResNetDenoiser(nn.Module):
    def __init__(self):
        super(ResNetDenoiser, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove classification layer

        # Custom denoising head
        self.denoise_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),  # Output RGB image
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.resnet(x)
        denoised = self.denoise_head(features.unsqueeze(-1).unsqueeze(-1))  # Expand dims
        return denoised

# Load model
model = ResNetDenoiser().to(device)
model.eval()

# Image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

# Denoising function
def apply_resnet_denoising(image: Image.Image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        denoised_tensor = model(image_tensor)
    denoised_image = transforms.ToPILImage()(denoised_tensor.squeeze(0))
    return denoised_image

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        denoised_image = apply_resnet_denoising(image)

        output_path = os.path.join(UPLOAD_DIR, f"denoised_{file.filename}")
        denoised_image.save(output_path)

        return JSONResponse(content={"filename": file.filename, "output": output_path, "download_url": f"/download/{output_path}"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    file_path = os.path.join(UPLOAD_DIR, file_path)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return JSONResponse(content={"error": "File not found"}, status_code=404)
