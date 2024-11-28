import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt

import utils.network as nt
from utils.dataloader import CropDataset

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# preprocess
transform = transforms.Compose([
    #transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# load data
data_dir = './data'
dataset = CropDataset(data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# load model
model = torch.load('model.pth')
# loss
criterion = nn.BCEWithLogitsLoss()

def produce_save(i):
    # the (i+1)th item
    for index, (image, label) in enumerate(train_loader):
        if index == i:
            image, label = image[1].to(device), label[1].to(device)
            break

    # forward
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
    loss = criterion(output, label)
    print(f'Loss: {loss.item()}')

    # binary
    output_binary = (output > 0.3).float()

    # numpy
    image_np = image.cpu().numpy().squeeze()
    label_np = label.cpu().numpy().squeeze()
    output_np = output.cpu().numpy().squeeze()
    output_binary_np = output_binary.cpu().numpy().squeeze()

    # figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].imshow(image_np, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(output_np, cmap='gray')
    axs[1].set_title('Model Output')
    axs[2].imshow(label_np, cmap='gray')
    axs[2].set_title('Label')
    for ax in axs:
        ax.axis('off')

    plt.savefig(f'demo/result{i}.png')

list_i = [327, 219, 355, 393]
for i in list_i:
    produce_save(i)
