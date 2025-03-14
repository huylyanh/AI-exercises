from ultralytics import YOLO
import torch
import os
from ultralytics import settings

model = YOLO("yolo11n.pt")  

if torch.cuda.is_available():
    device = 0  # Use GPU 0
    print("CUDA is available! Training on GPU.")
else:
    device = "cpu"  # Use CPU
    print("CUDA is NOT available. Training on CPU.")

project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

results = model.train(
    data=os.path.join(project_dir, "data", "football_dataset", "football.yaml"),
    epochs=20,
    batch=16,  
    imgsz=640,
    device=device,
    workers=4,
    project = os.path.join(project_dir, "runs") 
)




