from ultralytics import YOLO
import torch
import os
from ultralytics import settings

project_dir = os.path.dirname(os.path.abspath(__file__))
last_model_path = os.path.join(project_dir, "runs", "train6", "weights", "last.pt")
data_yaml_path = os.path.join(project_dir, "data", "football_dataset", "football.yaml")
project_name=os.path.join(project_dir, "runs")

num_epochs = 10

if not os.path.exists(last_model_path):
    print(f"Error: Model checkpoint not found at {last_model_path}.")
else:
    try:
        model = YOLO(last_model_path)  
    except Exception as e:
        print(f"Error loading model: {e}")

    if torch.cuda.is_available():
        device = 0  # Use GPU 0
        print("CUDA is available! Training on GPU.")
    else:
        device = "cpu"  # Use CPU
        print("CUDA is NOT available. Training on CPU.")

    try:
        results = model.train(
            data=data_yaml_path,
            epochs=num_epochs,
            batch=16,  
            imgsz=640,
            device=device,
            workers=4,
            project=project_name,
            resume = True
        )
        print("Training continues successfully.")
    except Exception as e:
        print(f"Error during training: {e}")




