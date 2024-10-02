import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

# Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset class for loading images and masks
class MRIDataset(Dataset):
    """
    A custom dataset class for loading MRI images and their corresponding masks.
    Attributes:
        image_dir (str): Directory containing the MRI images.
        mask_dir (str): Directory containing the mask images.
        transform (callable, optional): A function/transform to apply to both the image and the mask.
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(index): Returns the image and its corresponding mask at the specified index.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".tif", "_mask.tif")
        )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0).float()

        return image, mask


# U-Net Model Architecture
class UNet(nn.Module):
    """
    UNet model for image segmentation and classification.
    This model performs both segmentation and binary classification (e.g., cancer detection).
    It consists of an encoder-decoder architecture with skip connections and a classification head.
    Attributes:
        encoder1 (nn.Sequential): First encoder block.
        encoder2 (nn.Sequential): Second encoder block.
        encoder3 (nn.Sequential): Third encoder block.
        encoder4 (nn.Sequential): Fourth encoder block.
        bottleneck (nn.Sequential): Bottleneck layer.
        decoder4 (nn.Sequential): First decoder block.
        decoder3 (nn.Sequential): Second decoder block.
        decoder2 (nn.Sequential): Third decoder block.
        decoder1 (nn.Sequential): Fourth decoder block.
        final_conv (nn.Conv2d): Final convolutional layer for segmentation output.
        classifier (nn.Sequential): Classification head.
    Methods:
        encoder_block(in_channels, out_channels):
            Creates an encoder block with two convolutional layers followed by ReLU activations and max pooling.
        decoder_block(in_channels, out_channels):
            Creates a decoder block with a transposed convolutional layer followed by two convolutional layers and ReLU activations.
        forward(x):
            Forward pass through the network. Returns both segmentation and classification outputs.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
    Returns:
        segmentation_output (torch.Tensor): Segmentation output tensor of shape (batch_size, 1, height, width).
        classification_output (torch.Tensor): Classification output tensor of shape (batch_size, 2).
    """

    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.encoder_block(3, 64)
        self.encoder2 = self.encoder_block(64, 128)
        self.encoder3 = self.encoder_block(128, 256)
        self.encoder4 = self.encoder_block(256, 512)
        self.bottleneck = self.encoder_block(512, 1024)

        self.decoder4 = self.decoder_block(1024, 512)
        self.decoder3 = self.decoder_block(512, 256)
        self.decoder2 = self.decoder_block(256, 128)
        self.decoder1 = self.decoder_block(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(512, 2),  # Binary classification (cancer or no cancer)
        )

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        # Segmentation output
        segmentation_output = torch.sigmoid(self.final_conv(dec1))

        # Classification output
        classification_output = self.classifier(bottleneck)

        return segmentation_output, classification_output


def get_image_mask_path(data_path):
    image_file_paths = []
    mask_file_paths = []

    for file_names in os.walk(data_path):
        for file_name in file_names[2]:
            if "mask" in file_name:
                mask_file_paths.append(os.path.join(file_names[0], file_name))
                image_file_paths.append(
                    os.path.join(file_names[0], file_name.replace("_mask", ""))
                )

    return image_file_paths, mask_file_paths


def run_epoch(
    model,
    dataloader,
    criterion_segmentation,
    criterion_classification,
    num_epochs=20,
    optimizer=None,
):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            segmentation_outputs, classification_outputs = model(images)

            # Calculate segmentation loss
            loss_segmentation = criterion_segmentation(
                segmentation_outputs, masks.unsqueeze(1)
            )  # Add channel dimension for masks

            # Calculate classification loss (considering presence of cancer based on mask)
            cancer_labels = (
                (masks > 0).float().squeeze().long()
            )  # Convert mask to labels (0 or 1)
            loss_classification = criterion_classification(
                classification_outputs, cancer_labels
            )

            # Total loss
            loss = loss_segmentation + loss_classification
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}"
        )


# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    total_dice = 0
    total_iou = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            segmentation_outputs, classification_outputs = model(images)
            preds = (segmentation_outputs > 0.5).float()  # Thresholding

            # Calculate Dice and IoU
            dice = 2 * (preds * masks).sum() / (preds.sum() + masks.sum())
            total_dice += dice.item()

            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            iou = intersection / (
                union + 1e-6
            )  # Small constant to avoid division by zero
            total_iou += iou.item()

            # Classification accuracy
            cancer_labels = (
                (masks > 0).float().squeeze().long()
            )  # Convert mask to labels (0 or 1)
            _, predicted = torch.max(
                classification_outputs, 1
            )  # Get the class with the highest score
            total_correct += (predicted == cancer_labels).sum().item()
            total_samples += cancer_labels.size(0)

    return (
        total_dice / len(dataloader),
        total_iou / len(dataloader),
        total_correct / total_samples,
    )


if __name__ == "__main__":
    # Transformations
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    # Path
    data_path = "dataset\lgg-mri-segmentation\kaggle_3m"
    image_dir, mask_dir = get_image_mask_path(data_path)

    # Dataset and DataLoader
    dataset = MRIDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model Initialization
    model = UNet().to(device)

    # Loss Functions
    criterion_segmentation = nn.BCEWithLogitsLoss()
    criterion_classification = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 10
    run_epoch(
        model,
        dataloader,
        criterion_segmentation,
        criterion_classification,
        num_epochs,
        optimizer,
    )

    # Evaluate the model
    dice_score, iou_score, accuracy = evaluate_model(model, dataloader)
    print(
        f"Dice Score: {dice_score:.4f}, IoU Score: {iou_score:.4f}, Classification Accuracy: {accuracy:.4f}"
    )
