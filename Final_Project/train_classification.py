from dataset_classification import FootballDataset
from model_classification import MyCNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import ToTensor, Resize, Compose, Normalize

from tqdm.autonotebook import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
import os

def train():
    train_path = "data/football_train"
    val_path = "data/football_val"
    logging_path = "tensorboard"
    checkpoint_path = "classification_models"
    num_epochs = 50
    batch_size=16
    learning_rate = 1e-3
    momentum = 0.9
    num_classes = 12

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found.")
    else:
        device = torch.device("cpu")
        print("CUDA device not found. Training on CPU.")    

    model = MyCNN(num_classes=num_classes).to(device)

    best_acc = -1
    start_epoch = 0

    # Check for last checkpoint
    last_checkpoint_path = os.path.join(checkpoint_path, "last.pt")
    metadata_path = os.path.join(checkpoint_path, "metadata.pt") 
    if resume and os.path.exists(last_checkpoint_path):
        print(f"Resume training from: {last_checkpoint_path}")
        model.load_state_dict(torch.load(last_checkpoint_path))
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            start_epoch = metadata["epoch"]
            best_acc = metadata["best_acc"]
        else:
            print("Error: metadata file not found.")
    else:
        print("Starting training from scratch.")

    transform = Compose([
        ToTensor(), # Đưa kênh màu từ cuối lên đầu
        Resize((224, 224)),
    ])

    train_dataset = FootballDataset(root=train_path, transform=transform, max_players_per_frame=10)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_dataset = FootballDataset(root=val_path, transform=transform, max_players_per_frame=10)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    num_iters = len(train_dataloader)
    print("num of iterations: ", num_iters)

    if not os.path.isdir(logging_path):
        os.makedirs(logging_path)
    writer = SummaryWriter(logging_path)       

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    for epoch in range(start_epoch, num_epochs):
        # 1) Training stage
        model.train() # Chi cho mo hinh biet trong qua trong training, do co 1 so layers qua trinh train chay 1 kieu, qua trong val chay 1 kieu
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            # Forward
            images = images.to(device) # torch.Size([16, 10, 3, 224, 224])
            labels = labels.to(device) # torch.Size([16, 10])

            mask = (labels != -1)  # torch.Size([16, 10])

            # Reshape images and labels to match the model's expected input
            images_reshape = images.view(-1, 3, 224, 224) # torch.Size([160, 3, 224, 224])
            labels_reshape = labels.view(-1) # torch.Size([160])

            masked_images = images_reshape[mask.view(-1)] # torch.Size([x, 3, 224, 224])
            masked_labels = labels_reshape[mask.view(-1)] # torch.Size([x])

            if masked_images.shape[0] > 0:
                output = model(masked_images)                 
                
                loss_value = criterion(output, masked_labels)
                # print("Epoch {}/{}. Iter {}/{}. Loss {:0.4f}".format(epoch + 1, num_epochs, iter + 1, num_iters, loss_value))
                progress_bar.set_description(
                    "Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, num_epochs, loss_value))
                writer.add_scalar(tag="Train/Loss", scalar_value=loss_value, global_step=(epoch * num_iters) + iter)

                # Backward
                optimizer.zero_grad() # xóa gradiant đã tích lũy từ các iteration trước, làm sạch buffer
                loss_value.backward() #chạy optimizer để tinh gradient cho toan bo tham so cua mo hinh
                optimizer.step() # update parameters cua mo hinh dựa vào gradient tính ở bước trên
        
        # 2) Validation stage
        model.eval() # 1 vài layers sẽ hoạt động khác đi so với quá trình train: dropout, batchnorm
        val_losses = []

        val_predictions = []
        val_labels = []

        progress_bar = tqdm(val_dataloader, colour="yellow")

        # with torch.inference_mode: # from pytorch 1.9
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                # print(epoch, images.shape, labels.shape)

                # Forward
                images = images.to(device) # torch.Size([16, 10, 3, 224, 224])
                labels = labels.to(device) # torch.Size([16, 10])

                mask = (labels != -1) # torch.Size([16, 10])

                images_reshape = images.view(-1, 3, 224, 224) # torch.Size([160, 3, 224, 224])
                labels_reshape = labels.view(-1) # torch.Size([160])

                masked_images = images_reshape[mask.view(-1)] # torch.Size([x, 3, 224, 224])
                masked_labels = labels_reshape[mask.view(-1)] # torch.Size([x])

                if masked_images.shape[0] > 0:
                    output = model(masked_images) 

                    predictions = torch.argmax(output, dim=1)
                    
                    loss_value = criterion(output, masked_labels) #loss_value là tensor kiểu scalar (0 chiều) = 1 value
                    val_losses.append(loss_value.item())

                    val_labels.extend(masked_labels.cpu().tolist())
                    val_predictions.extend(predictions.cpu().tolist())

        avg_loss = np.mean(val_losses)
        acc = accuracy_score(val_labels, val_predictions)
        print("Accuracy: {}. Average loss: {}".format(acc, avg_loss))

        writer.add_scalar(tag="Val/Loss", scalar_value=avg_loss, global_step=epoch)
        writer.add_scalar(tag="Val/Accuracy", scalar_value=acc, global_step=epoch)

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "last.pt"))
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "best.pt"))
            best_acc = acc

        # Save metadata
        metadata = {
            "epoch": epoch + 1,
            "best_acc": best_acc
        }
        torch.save(metadata, metadata_path)
       
if __name__ == "__main__":
    resume = True
    train()