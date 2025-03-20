from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import os
import shutil
from dip_model import process_image  # Ensure this imports your DIP model function

app = FastAPI()

# Allow requests from any host (important for deployment)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Create an uploads folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "DIP FastAPI server is running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image using DIP
    output_path = os.path.join(UPLOAD_FOLDER, "denoised_" + file.filename)
    process_image(file_path, output_path)  # Call DIP processing function

    return {
        "filename": file.filename,
        "output": output_path,
        "download_url": f"/download/{os.path.basename(output_path)}"
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)  # Increased timeout
