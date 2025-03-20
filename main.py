import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# Define a ResNet-based DIP Model
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x  # Skip connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Add skip connection
        return self.relu(out)

# Define the full DIP Model with ResNet blocks
class ResNetDIP(nn.Module):
    def __init__(self):
        super(ResNetDIP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNetBlock(64, 64),
            ResNetBlock(64, 64)
        )
        self.decoder = nn.Sequential(
            ResNetBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the model
model = ResNetDIP()
model.eval()

# FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "DIP FastAPI Running with ResNet!"}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Increased resolution
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    output_img = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1))  # Fix scaling
    output_path = "uploads/denoised_" + file.filename
    output_img.save(output_path)

    return {"filename": file.filename, "output": output_path, "download_url": f"/download/{output_path}"}
