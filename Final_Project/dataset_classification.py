from torch.utils.data import Dataset
import os
from typing import List, Tuple
import json
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor, Resize, Compose

class FootballDataset(Dataset):
    def __init__(self, root, transform, max_players_per_frame = 10):
        print("*** Enter init FootballDataset")
        super().__init__()

        sub_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        self.transform = transform
        self.frame_infos: List[Tuple[str, List[dict]]] = []
        self.video_frame_counts: List[int] = []
        self.max_players_per_frame = max_players_per_frame
    
        total_frame_count = 0
        print("len subdirs: ", len(sub_dirs))
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(root, sub_dir)
            print("Sub dir: ", sub_dir_path)

            video_path = None
            json_path = None
            if os.path.isdir(sub_dir_path):
                for file in os.listdir(sub_dir_path):
                    if file.endswith(".mp4"):
                        video_path = os.path.join(sub_dir_path, file)
                    elif file.endswith(".json"):
                        json_path = os.path.join(sub_dir_path, file)
            
            if video_path and json_path:
                cap = cv2.VideoCapture(video_path)
                video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_frame_counts.append(video_frame_count)
                cap.release()

                data = None
                with open(json_path, "r") as f:
                    data = json.load(f)
            
                if data:
                    annotations_list = [[] for _ in range(video_frame_count)]
                    for annotation in data["annotations"]:
                        if annotation["category_id"] == 4:
                            image_id = annotation["image_id"] - 1
                            annotations_list[image_id].append(annotation)

                    for annotation in annotations_list:
                        self.frame_infos.append((video_path, annotation))
                    
                    total_frame_count += len(annotations_list)

        self.total_frame_count = total_frame_count
        print("*** Out init FootballDataset - Total frame count: ", total_frame_count)
                    
    def __len__(self):
        return self.total_frame_count

    def __getitem__(self, idx):
        video_path, annotation_list = self.frame_infos[idx]
        player_images = []
        player_labels = []

        if annotation_list:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Could not open video: ", video_path)
            else:
                frame_index_in_video = 0
                current_frame_index = 0
                for num_frames in self.video_frame_counts:
                    current_frame_index += num_frames
                    if idx < current_frame_index:
                        frame_index_in_video = idx - (current_frame_index - num_frames)
                        break

                for _ in range(frame_index_in_video):
                    ret, frame = cap.read()

                ret, frame = cap.read()
                cap.release()

                if ret:
                     # Extract player images and labels
                    for annotation in annotation_list:
                        bbox = annotation["bbox"]
                        x_min, y_min, width, height = bbox
                        x_max = x_min + width
                        y_max = y_min + height

                        player_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if player_image.size > 0:
                            if self.transform:
                                player_image = self.transform(player_image)

                            player_images.append(player_image)

                            player_label = None
                            number_visible = None
                            if "attributes" in annotation:
                                if "jersey_number" in annotation["attributes"]:
                                    player_label =  int(annotation["attributes"]["jersey_number"])

                                if "number_visible" in annotation["attributes"]:
                                    number_visible = annotation["attributes"]["number_visible"]

                                if number_visible in ["invisible"]:
                                    player_label = 0
                                elif player_label >= 11:
                                    player_label = 11

                            player_labels.append(player_label)
                    
                    num_players = len(player_images)
                    if num_players < self.max_players_per_frame:
                        empty_image = torch.zeros_like(player_images[0]) if player_images else torch.zeros(3, 224, 224)
                        empty_label = -1

                        padding_size = self.max_players_per_frame - num_players
                        player_images.extend(empty_image for _ in range(padding_size))
                        player_labels.extend(empty_label for _ in range(padding_size))
                    elif num_players > self.max_players_per_frame:
                        player_images = player_images[:self.max_players_per_frame]
                        player_labels = player_labels[:self.max_players_per_frame]

        # return batches of images and labels
        return torch.stack(player_images), torch.tensor(player_labels)

if __name__ == '__main__':
    dataset_path = "data/football_train"

    transform = Compose([
        ToTensor(), 
        Resize((224, 224))
    ])

    dataset = FootballDataset(root=dataset_path, transform=transform)
    print("Dataset length: ", len(dataset))

    images, labels = dataset[500]
    print("Images: ", len(images))
    print("Labels: ", len(labels))
    print(images.shape)
    print(labels.shape)

    for i, label in enumerate(labels):
        print("label {}: {}".format(i, label))
        if len(images.shape) > 0:
            # rearrange the dimensions of the image tensor: from (channel, height, width) to (height, width, channel)
            # Convert from pytorch tensor to a numpy array
            # OpenCV reads and stores images in BGR format, matplotlib expects images to be in RGB format.
            plt.imshow(cv2.cvtColor(images[i].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB))
            plt.show()




