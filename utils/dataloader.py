import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    '''
    CustomDataset: A custom dataset class for loading images and labels
    '''
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for i in range(8):
                    image_path = os.path.join(folder_path, f'Part{i}.jpg')
                    label_path = os.path.join(folder_path, f'Part{i}_label.bmp')
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        self.image_paths.append(image_path)
                        self.label_paths.append(label_path)
                    else:
                        print(f"Missing image or label: {image_path}, {label_path}")

        print("Data loaded")
        print(f"Found {len(self.image_paths)} images and {len(self.label_paths)} labels")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = Image.open(self.label_paths[idx]).convert('L')
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            image = normalize(image)
        else:
            transform = transforms.ToTensor()
            image = transform(image)
            label = transform(label)
        return image, label

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_label_path(self, idx):
        return self.label_paths[idx]

class CropDataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for i in range(8):
                    image_path = os.path.join(folder_path, f'Part{i}.jpg')
                    label_path = os.path.join(folder_path, f'Part{i}_label.bmp')
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        self.image_paths.append(image_path)
                        self.label_paths.append(label_path)
                    else:
                        print(f"Missing image or label: {image_path}, {label_path}")

        print("Data loaded")
        print(f"Found {len(self.image_paths)} images and {len(self.label_paths)} labels")

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image_array = np.array(image)
        normalized_array = image_array / 255.0
        image = Image.fromarray((normalized_array * 255).astype(np.uint8))
        label = Image.open(self.label_paths[idx]).convert('L')
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # 裁剪图片
        cropped_images = []
        cropped_labels = []
        start_x = 170
        start_y = 0
        step_y = 400  # 每次移动360像素

        for i in range(3):
            box = (start_x, start_y + i * step_y, start_x+ 320, start_y + i * step_y + 320)
            cropped_image = image.crop(box)
            cropped_label = label.crop(box)
            cropped_images.append(cropped_image)
            cropped_labels.append(cropped_label)

        # 应用变换
        if self.transform:
            cropped_images = [self.transform(img) for img in cropped_images]
            cropped_labels = [self.transform(lbl) for lbl in cropped_labels]
            cropped_images = [normalize(img) for img in cropped_images]
        else:
            transform = transforms.ToTensor()
            cropped_images = [transform(img) for img in cropped_images]
            cropped_labels = [transform(lbl) for lbl in cropped_labels]
            cropped_images = [self.transform(img) for img in cropped_images]

        return cropped_images, cropped_labels

    def __len__(self):
        return len(self.image_paths)
