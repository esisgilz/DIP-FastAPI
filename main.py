from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os

# Define FastAPI app
app = FastAPI()

# Load ResNet model (modify as needed for DIP-based denoising)
class ResNetDenoiser(torch.nn.Module):
    def __init__(self):
        super(ResNetDenoiser, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove classification head
    
    def forward(self, x):
        return self.model(x)

denoiser = ResNetDenoiser()
denoiser.eval()

# Transformation for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output_tensor = denoiser(input_tensor)
        
        output_image = transforms.ToPILImage()(output_tensor.squeeze())
        output_path = os.path.join(PROCESSED_FOLDER, "denoised.png")
        output_image.save(output_path)
        
        return {"message": "Processing complete", "output_file": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")
