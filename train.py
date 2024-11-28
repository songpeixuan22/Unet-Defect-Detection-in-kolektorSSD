import torch,gc
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import utils.network as nt
from utils.trainer import Trainer
from utils.dataloader import CropDataset
from utils.loss import DiceBCELoss

gc.collect()
torch.cuda.empty_cache()

# hyperparameters
lr = 1e-4
epochs = 250
batch_size = 32
weight_decay = 0
retrain = 0
# the lr changing
stepsize = 15
gamma = 0.5

# create model & citerion
# model = nt.U_Net(img_ch=1, output_ch=1)
# model = nt.U_Net_(img_ch=1, output_ch=1)
# model = nt.ResU_Net(img_ch=1, output_ch=1)
# model = nt.AttU_Net(img_ch=1, output_ch=1)
model = nt.ResAttU_Net_(img_ch=1, output_ch=1)
# criterion = DiceBCELoss()
criterion = nn.BCEWithLogitsLoss()

# create SummaryWriter
logdir = "log/w12/" + 'net:ResAttU_ ' + 'retrain:01 ' + 'loss:bce '+ 'decay:(1e-4,15,0.5) '+ 'bs:64 '
writer = SummaryWriter(logdir)

# transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
#    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# data set
data_dir = './data'
dataset = CropDataset(data_dir, transform=transform)
print(f'Loaded {len(dataset)} samples')
# divide dataset into training and testing sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# set device & parallel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = list(range(torch.cuda.device_count()))
model = nn.DataParallel(model, device_ids=device_ids).to(device)
print(f'Using devices: {device_ids}')

# optimizer and criterion
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

if retrain:
    model = torch.load('model.pth')
else:
    model.to(device)

# create trainer and train the model
print(f'Learning rate: {lr}; Batch size: {batch_size}; Epochs: {epochs}')
trainer = Trainer(model, optimizer, criterion, device, writer)
trainer.train(train_loader, test_loader, epochs=epochs)

# save the model
torch.save(model, 'model.pth')

# close the SummaryWriter
writer.close()
