import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

class Trainer:
    '''
    Trainer: A class for training a model
    '''
    def __init__(self, model, optimizer, criterion, device, writer=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def train(self, train_loader, val_loader, epochs):
        scaler = GradScaler('cuda')
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
            for i, (x_list, y_list) in progress_bar:
                for x, y in zip(x_list, y_list):
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    with autocast('cuda'):
                        output = self.model(x)
                        # mean_num = torch.mean(output)
                        # output = torch.div(output, mean_num)
                        loss = self.criterion(output, y)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    running_loss += loss.item() / len(x_list)

                progress_bar.set_postfix(loss=running_loss/(i+1))

            running_loss /= len(train_loader)
            if self.writer:
                self.writer.add_scalar('training loss', running_loss, epoch)
            
            # 验证步骤
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_list, y_list in val_loader:
                    for x, y in zip(x_list, y_list):
                        x, y = x.to(self.device), y.to(self.device)
                        with autocast('cuda'):
                            output = self.model(x)
                            loss = self.criterion(output, y)
                        val_loss += loss.item() / len(x_list)
            
            val_loss /= len(val_loader)
            if self.writer:
                self.writer.add_scalar('validation loss', val_loss, epoch)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss}, Validation Loss: {val_loss}')
            print()
