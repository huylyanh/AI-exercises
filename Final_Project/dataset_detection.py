import os
import cv2
import json
import random
import shutil
from typing import Final

input_dataset_path = "data/football_train"
output_dataset_path = "data/football_dataset"
MAX_FRAMES: Final = 0

def main():
    sub_dirs = [d for d in os.listdir(input_dataset_path) if os.path.isdir(os.path.join(input_dataset_path, d))]

    all_images_dir = os.path.join(output_dataset_path, "all_images")
    all_labels_dir = os.path.join(output_dataset_path, "all_labels")
    os.makedirs(all_images_dir, exist_ok=True)
    os.makedirs(all_labels_dir, exist_ok=True)

    frame_count = 0
    image_files = []
    for sub_dir in sub_dirs:
        subdir_path = os.path.join(input_dataset_path, sub_dir)
        print("***subdir_path: ", subdir_path)

        video_file_path = None
        json_file_path = None

        for file in os.listdir(subdir_path):
            if file.endswith(".mp4"):
                video_file_path = os.path.join(subdir_path, file)
            elif file.endswith(".json"):
                json_file_path = os.path.join(subdir_path, file)

        data = None
        if json_file_path:
            try:
                with open(json_file_path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print("Json file not found at {}".format(json_file_path))
                continue
            except json.JSONDecodeError:
                print("Invalid Json file at {}".format(json_file_path))
                continue
            
        if video_file_path:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                print("Could not open video at {}".format(video_file_path))
            else:
                no_frames_in_video  = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: 
                        break

                    frame_count += 1
                    no_frames_in_video += 1
                    frame_filename = "frame_{:06d}.jpg".format(frame_count)
                    frame_filepath = os.path.join(all_images_dir, frame_filename)
                    label_filename = frame_filename.replace(".jpg", ".txt")
                    label_filepath = os.path.join(all_labels_dir, label_filename)

                    if os.path.exists(label_filepath):
                        os.remove(label_filepath)

                    has_ball_or_player = False
                    if data:
                        image_id = None
                        for image in data["images"]:
                            if no_frames_in_video == image["id"]:
                                image_id = image["id"]
                                image_width = image["width"]
                                image_height = image["height"]
                                break
                        
                        if image_id is not None:
                            for annotation in data["annotations"]:
                                if image_id == annotation["image_id"]:
                                    category_id = annotation["category_id"]
                                    if category_id in [3, 4]:
                                        has_ball_or_player = True
                                        bbox = annotation["bbox"]
                                        
                                        x_min, y_min, width, height = bbox
                                        x_center = x_min + width / 2
                                        y_center = y_min + height / 2

                                        x_center /= image_width
                                        y_center /= image_height
                                        width /= image_width
                                        height /= image_height
                                    
                                        with open(label_filepath, "a") as f:
                                            f.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f}".format(category_id - 3, x_center, y_center, width, height))
                                            f.write("\n")
                    
                            if has_ball_or_player:
                                cv2.imwrite(frame_filepath, frame)
                                image_files.append(frame_filename)

                                if MAX_FRAMES  > 0 and frame_count >= MAX_FRAMES: 
                                    break

            print("Total frames: ", no_frames_in_video)
            cap.release()

        if MAX_FRAMES  > 0 and frame_count >= MAX_FRAMES: 
            break

    train_val_ratio = 0.8
    image_train_dir = os.path.join(output_dataset_path, "images", "train")
    image_val_dir = os.path.join(output_dataset_path, "images", "val")
    label_train_dir = os.path.join(output_dataset_path, "labels", "train")
    label_val_dir = os.path.join(output_dataset_path, "labels", "val")

    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    random.shuffle(image_files)

    train_size = int(len(image_files) * train_val_ratio)
    train_image_files = image_files[:train_size]
    val_image_files = image_files[train_size:]

    for file in train_image_files:
        shutil.copy(os.path.join(all_images_dir, file), image_train_dir)
        label_file_name = file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(all_labels_dir, label_file_name), label_train_dir)

    for file in val_image_files:
        shutil.copy(os.path.join(all_images_dir, file), image_val_dir)
        label_file_name = file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(all_labels_dir, label_file_name), label_val_dir)

    yaml_file_name = "football.yaml"
    yaml_file_path = os.path.join(output_dataset_path, yaml_file_name)
    num_classes = 2
    name_classes = ["ball", "player"]

    with open(yaml_file_path, "w") as f:
        f.write(f"path: {output_dataset_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {name_classes}\n")

if __name__ == "__main__":
    main()


            










         
         
   



