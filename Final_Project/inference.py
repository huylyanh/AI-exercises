import os
import shutil
from ultralytics import YOLO
import cv2
from torchvision.transforms import Compose, ToTensor, Resize
import torch
from model_classification import MyCNN


project_dir = os.path.dirname(os.path.abspath(__file__))

# Detection model paths
detection_model_path = os.path.join(project_dir, "runs", "train6", "weights", "best.pt")
input_video_path = os.path.join(project_dir, "data", "Match_2031_5_0_test", "Match_2031_5_0_test.mp4")
output_dir = os.path.join(project_dir, "output_annotated")

# Classification model paths
classification_model_path = os.path.join(project_dir, "classification_models", "best.pt")

# Confidence thresholds
detection_confidence_ts = 0.6
classification_confidence_ts = 0.6

# Classification transform
classification_transform = Compose([
    ToTensor(),
    Resize((224, 224))
])

if not os.path.exists(detection_model_path):
    print("Error: Detection model not found")
    exit()

if not os.path.exists(classification_model_path):
    print("Error: Classification model not found")
    exit()

try:
    detection_model = YOLO(detection_model_path)
except Exception as e:
    print(f"Error loading detection model: {e}")
    exit()

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model = MyCNN(num_classes=12)
    classification_model.load_state_dict(torch.load(classification_model_path, map_location="cpu"))
    classification_model.to(device)
    classification_model.eval()
except Exception as e:
    print(f"Error loading classification model: {e}")
    exit()

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

try:
    cap = cv2.VideoCapture(input_video_path)
except Exception as e:
    print(f"Error opening video file: {e}")
    exit()

if not cap.isOpened():
    print("Could not open video at {}".format(input_video_path))
    exit()

print("Num of frames: ", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
os.makedirs(output_dir, exist_ok=True)

# Get video properties for the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter 
video_output_file_path = os.path.join(output_dir, "annotated_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_file_path, fourcc, fps, (frame_width, frame_height))

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale_detection = 0.7 
font_thickness_detection = 2
font_scale_classification = 1.5
font_thickness_classification = 3 

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    annotated_frame = frame.copy()

    detection_outputs = detection_model .predict(source=frame, conf=detection_confidence_ts, verbose=False)

    for r in detection_outputs:
        boxes = r.boxes
        for box in boxes:
            bb = box.xyxy[0].int().tolist()
            conf = round(box.conf[0].item(), 2)
            class_id = int(box.cls[0].item())
            class_name = detection_model.names[class_id]

            if class_name == "player":
                player_image = frame[bb[1]:bb[3], bb[0]:bb[2]]

                if player_image.size > 0: 
                    player_image = classification_transform(player_image)
                    player_image = player_image.unsqueeze(0).to(device)

                    with torch.no_grad():
                        classification_output = classification_model(player_image)
                        predicted_class = torch.argmax(classification_output).item()

                    player_number = predicted_class
                    if player_number == 0:
                        player_number_description = "Invisible"
                    elif player_number == 11:
                        player_number_description = "> 10"
                    else:
                        player_number_description = "{}".format(player_number)

                    # Drawing bounding box for player
                    # object_image = frame[b[1]:b[3], b[0]:b[2]]
                    cv2.rectangle(annotated_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

                    # annotate detection 
                    label_text_detection = "{}: {:.2f}".format(class_name, conf)
                    cv2.putText(annotated_frame, label_text_detection, (bb[0], bb[1] - 30), font, font_scale_detection, (0, 255, 0), font_thickness_detection)

                    # annotate classification
                    label_text_classification = "{}".format(player_number_description)
                    text_size = cv2.getTextSize(label_text_classification, font, font_scale_classification, font_thickness_classification)[0]
                    text_x = bb[0]
                    text_y = bb[3] + text_size[1] + 10
                    cv2.putText(annotated_frame, label_text_classification, (text_x, text_y), font, font_scale_classification, (0, 0, 255), font_thickness_classification)

            # class_dir = os.path.join(output_dir, class_name)
            # os.makedirs(class_dir, exist_ok=True)

    # frame_filename = "frame_{:06d}.jpg".format(frame_count)
    # frame_filepath = os.path.join(output_dir, frame_filename)
    # cv2.imwrite(frame_filepath, annotated_frame)

    out.write(annotated_frame)
    print("Annotated frame {} is written to video".format(frame_count))

    if frame_count == 1000:
        break

cap.release()
out.release()
print("Annotated video is saved to ".format(video_output_file_path))








            









    



