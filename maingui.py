import os
import sys
import gc
import time
from datetime import timedelta, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.color import lab2xyz, xyz2rgb
from torchvision import transforms
import logging
import random
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, 
    QLineEdit, QMessageBox, QProgressBar, QTextEdit, QComboBox, QHBoxLayout, QSplitter, QGroupBox,
    QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
import threading
from tqdm import tqdm
import glob
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from themes.themesgui import THEMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

HYPERPARAMS = {
    "lambda_L1": 100.0,
    "lr_G": 2e-4,
    "lr_D": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,
    "dropout_rate": 0.3,
    "batch_size": 2,
    "n_workers": 4,
    "pin_memory": True,
    "image_size": 512,
    "generator_train_epochs": 10,
    "gan_train_epochs": 100,
    "warmup_epochs": 5,
    "patience": 5,
    "gradient_accumulation_steps": 4,
    "weight_decay": 2e-5,
    "initial_lr": 1e-4,
    "train_split": 0.99,
    "val_split": 0.01,
    "cosine_annealing_eta_min": 0,
    "cosine_annealing_T_max": 200,
    "nickname": "nitrocon",
    "model_name": "Supercolor",
    "version": "00a",
    "gan_mode": "vanilla",
    "real_label": 1.0,
    "fake_label": 0.0,
    "optimizer_betas": (0.5, 0.999),
    "num_filters": 64,
    "n_down": 3,
    "brightness_range": (-0.02, 0.02),
    "contrast_range": (-0.03, 0.02),
    "saturation_range": (1.00, 1.05),
    "distortion_scale": 0.1,
    "gaussian_blur_kernel_size": 3,
    "gaussian_blur_sigma": (0.1, 2.0),
    "random_rotation_degrees": 45,
    "random_perspective_p": 0.5,
    "random_horizontal_flip_p": 0.5,
    "num_augmentations": 1,
    "app_width": 1250,
    "app_height": 565,
    "gui1_width": 220,
    "gui2_width": 270,
    "infobox1_width": 250,
    "epochs_groupbox_height": 60,
    "groupbox_margins": 5,
    "widget_spacing": 0,
    "widget_height": 50,
    "grad_scaler_enabled": True,
    "grad_scaler_init_scale": 65536.0,
    "grad_scaler_growth_factor": 2.0,
    "grad_scaler_backoff_factor": 0.5,
    "grad_scaler_growth_interval": 2000,
    "grad_scaler_enabled_for_backward": True,
    "supported_image_extensions": ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'],
    "max_image_length": 25,
    "save_dir": "models",
    "colorized_folder_name": "colorized",
}

def get_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        cuda_version = torch.version.cuda
        cuda_enabled = torch.cuda.current_device() >= 0
        device = torch.device("cuda")
        return device, f"CUDA is available: {cuda_enabled}. Device: {device_name}, CUDA Version: {cuda_version}"
    else:
        device = torch.device("cpu")
        torch.set_num_threads(torch.get_num_threads())
        return device, f"CUDA is not available. Using CPU with {torch.get_num_threads()} threads."

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

class GANLoss(nn.Module):
    def __init__(self, gan_mode=HYPERPARAMS["gan_mode"], real_label=HYPERPARAMS["real_label"], fake_label=HYPERPARAMS["fake_label"]):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=HYPERPARAMS["num_filters"], n_down=HYPERPARAMS["n_down"]):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class MainModel(nn.Module):
    def __init__(self, architecture="resnet18", net_G=None, lr_G=HYPERPARAMS["lr_G"], lr_D=HYPERPARAMS["lr_D"], 
                 beta1=HYPERPARAMS["beta1"], beta2=HYPERPARAMS["beta2"], lambda_L1=HYPERPARAMS["lambda_L1"], 
                 nickname=HYPERPARAMS["nickname"], model_name=HYPERPARAMS["model_name"], version=HYPERPARAMS["version"]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.epoch = 0
        self.batch_idx = 0
        self.nickname = nickname
        self.model_name = model_name
        self.version = version
        self.architecture = architecture
        
        if net_G is None:
            if architecture == "resnet18":
                from architectures.dynamicunet.resnet18 import ResNet18DynamicUnetColorizationModel
                net_G = ResNet18DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"])
            elif architecture == "resnet34":
                from architectures.dynamicunet.resnet34 import ResNet34DynamicUnetColorizationModel
                net_G = ResNet34DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"])
            elif architecture == "resnet50":
                from architectures.dynamicunet.resnet50 import ResNet50DynamicUnetColorizationModel
                net_G = ResNet50DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"])
            elif architecture == "efficientnetb0":
                from architectures.dynamicunet.efficientnetb0 import EfficientNetB0DynamicUnet
                net_G = EfficientNetB0DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif architecture == "shufflenetv2x0_5":
                from architectures.dynamicunet.shufflenetv2x05 import ShuffleNetV2x05DynamicUnet
                net_G = ShuffleNetV2x05DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif architecture == "shufflenetv2x1_0":
                from architectures.dynamicunet.shufflenetv2x1_0 import ShuffleNetV2x1_0DynamicUnet
                net_G = ShuffleNetV2x1_0DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif architecture == "shufflenetv2x1_5":
                from architectures.dynamicunet.shufflenetv2x1_5 import ShuffleNetV2x1_5DynamicUnet
                net_G = ShuffleNetV2x1_5DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif architecture == "shufflenetv2x2_0":
                from architectures.dynamicunet.shufflenetv2x2_0 import ShuffleNetV2x2_0DynamicUnet
                net_G = ShuffleNetV2x2_0DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
        
        self.net_G = net_G.to(self.device)
        
        self.net_G.discriminator = PatchDiscriminator(input_c=3).to(self.device)
        
        self.GANcriterion = GANLoss(gan_mode=HYPERPARAMS["gan_mode"]).to(self.device)
        self.L1criterion = nn.L1Loss()
        
        self.opt_G = optim.Adam(
            self.net_G.parameters(), 
            lr=lr_G, 
            betas=(beta1, beta2), 
            weight_decay=HYPERPARAMS["weight_decay"]
        )
        self.opt_D = optim.Adam(
            self.net_G.discriminator.parameters(), 
            lr=lr_D, 
            betas=(beta1, beta2), 
            weight_decay=HYPERPARAMS["weight_decay"]
        )

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_G.discriminator(fake_image.detach())
        loss_D_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_G.discriminator(real_image)
        loss_D_real = self.GANcriterion(real_preds, True)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, loss_D_fake, loss_D_real

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_G.discriminator(fake_image)
        loss_G_GAN = self.GANcriterion(fake_preds, True)

        loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        loss_G = loss_G_GAN + loss_G_L1

        return loss_G, loss_G_GAN, loss_G_L1

    def optimize(self, data):
        self.setup_input(data)
        self.forward()

        self.set_requires_grad(self.net_G.discriminator, True)
        self.opt_D.zero_grad()
        loss_D, loss_D_fake, loss_D_real = self.backward_D()
        loss_D.backward()
        self.opt_D.step()

        self.set_requires_grad(self.net_G.discriminator, False)
        self.opt_G.zero_grad()
        loss_G, loss_G_GAN, loss_G_L1 = self.backward_G()
        loss_G.backward()
        self.opt_G.step()

        return loss_G, loss_G_GAN, loss_G_L1, loss_D, loss_D_fake, loss_D_real

def load_image_paths(dataset_dir):
    image_extensions = HYPERPARAMS["supported_image_extensions"]
    paths = []
    for ext in image_extensions:
        paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
    return paths

class AdjustBrightnessContrastSaturation:
    def __init__(self, brightness_range=HYPERPARAMS["brightness_range"], contrast_range=HYPERPARAMS["contrast_range"], saturation_range=HYPERPARAMS["saturation_range"]):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    def __call__(self, img):
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        img = transforms.functional.adjust_brightness(img, 1.0 + brightness_factor)

        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        img = transforms.functional.adjust_contrast(img, 1.0 + contrast_factor)

        saturation_factor = random.uniform(self.saturation_range[0], self.saturation_range[1])
        img = transforms.functional.adjust_saturation(img, saturation_factor)

        return img

class RandomResize:
    def __init__(self, image_size):
        self.image_size = image_size
        self.sizes = [int(image_size / (2 ** i)) for i in list(range(int(np.log2(image_size / 16)))) + [0]]
        self.sizes.append(16)

    def __call__(self, img):
        new_size = random.choice(self.sizes)
        img = transforms.functional.resize(img, (new_size, new_size))
        img = transforms.functional.resize(img, (self.image_size, self.image_size))
        return img

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', device=None, num_augmentations=HYPERPARAMS["num_augmentations"]):
        self.num_augmentations = num_augmentations
        self.initial_transforms = transforms.Compose([
            transforms.Resize((HYPERPARAMS["image_size"], HYPERPARAMS["image_size"])),
            transforms.RandomApply([RandomResize(HYPERPARAMS["image_size"])], p=0.75),
            transforms.Resize((HYPERPARAMS["image_size"], HYPERPARAMS["image_size"])),
        ])
        if split == 'train':
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomApply([AdjustBrightnessContrastSaturation()], p=0.5),
                transforms.RandomHorizontalFlip(p=HYPERPARAMS["random_horizontal_flip_p"]),
                transforms.RandomApply([transforms.RandomRotation(HYPERPARAMS["random_rotation_degrees"])], p=0.5),
                transforms.RandomPerspective(distortion_scale=HYPERPARAMS["distortion_scale"], p=HYPERPARAMS["random_perspective_p"]),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=HYPERPARAMS["gaussian_blur_kernel_size"], sigma=HYPERPARAMS["gaussian_blur_sigma"])], p=0.5),
            ])
        elif split == 'val':
            self.augmentation_transforms = transforms.Compose([])
        self.split = split
        self.size = HYPERPARAMS["image_size"]
        self.paths = paths
        self.device = device
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx // self.num_augmentations]).convert("RGB")
        img = self.initial_transforms(img)
        img = self.augmentation_transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return {'L': L.to(self.device), 'ab': ab.to(self.device)}
    
    def __len__(self):
        return len(self.paths) * self.num_augmentations

def make_dataloaders(batch_size=HYPERPARAMS["batch_size"], n_workers=HYPERPARAMS["n_workers"], pin_memory=HYPERPARAMS["pin_memory"], **kwargs):
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory, shuffle=True)
    return dataloader

class DataPreparationThread(QThread):
    data_prepared = pyqtSignal(list, list)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        paths = load_image_paths(self.folder_path)
        rand_idxs = np.random.permutation(len(paths))
        train_idxs = rand_idxs[:int(HYPERPARAMS["train_split"] * len(paths))]
        val_idxs = rand_idxs[int(HYPERPARAMS["train_split"] * len(paths)):]
        train_paths = [paths[i] for i in train_idxs]
        val_paths = [paths[i] for i in val_idxs]
        self.data_prepared.emit(train_paths, val_paths)

class ModelLoadingThread(QThread):
    model_loaded = pyqtSignal(object)

    def __init__(self, file_path, device):
        super().__init__()
        self.file_path = file_path
        self.device = device

    def run(self):
        try:
            checkpoint = torch.load(self.file_path, weights_only=True)
            architecture = checkpoint.get('architecture', 'resnet18')
            nickname = checkpoint.get('nickname', HYPERPARAMS["nickname"])
            model_name = checkpoint.get('model_name', HYPERPARAMS["model_name"])
            version = checkpoint.get('version', HYPERPARAMS["version"])
            
            if architecture == "resnet18":
                from architectures.dynamicunet.resnet18 import ResNet18DynamicUnetColorizationModel
                net_G = ResNet18DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            elif architecture == "resnet50":
                from architectures.dynamicunet.resnet50 import ResNet50DynamicUnetColorizationModel
                net_G = ResNet50DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            elif architecture == "efficientnetb0":
                from architectures.dynamicunet.efficientnetb0 import EfficientNetB0DynamicUnet
                net_G = EfficientNetB0DynamicUnet(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            elif architecture == "resnet34":
                from architectures.dynamicunet.resnet34 import ResNet34DynamicUnetColorizationModel
                net_G = ResNet34DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            elif architecture == "shufflenetv2x0_5":
                from architectures.dynamicunet.shufflenetv2x05 import ShuffleNetV2x05DynamicUnet
                net_G = ShuffleNetV2x05DynamicUnet(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            elif architecture == "shufflenetv2x1_0":
                from architectures.dynamicunet.shufflenetv2x1_0 import ShuffleNetV2x1_0DynamicUnet
                net_G = ShuffleNetV2x1_0DynamicUnet(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            elif architecture == "shufflenetv2x1_5":
                from architectures.dynamicunet.shufflenetv2x1_5 import ShuffleNetV2x1_5DynamicUnet
                net_G = ShuffleNetV2x1_5DynamicUnet(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            elif architecture == "shufflenetv2x2_0":
                from architectures.dynamicunet.shufflenetv2x2_0 import ShuffleNetV2x2_0DynamicUnet
                net_G = ShuffleNetV2x2_0DynamicUnet(self.device, size=HYPERPARAMS["image_size"]).to(self.device)
            else:
                self.model_loaded.emit(None)
                return

            model = MainModel(net_G=net_G, nickname=nickname, model_name=model_name, version=version, architecture=architecture)
            model.net_G.load_state_dict(checkpoint['model_state_dict'])
            model.net_G.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            model.opt_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            model.opt_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.model_loaded.emit(model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded.emit(None)

class GeneratorTrainingThread(QThread):
    update_progress = pyqtSignal(int, str, str, str, str, str, str)
    generator_training_finished = pyqtSignal(object)

    def __init__(self, net_G, train_dl, device, max_epochs, version, nickname, model_name, architecture, text_area, warmup_epochs):
        super().__init__()
        self.net_G = net_G
        self.train_dl = train_dl
        self.device = device
        self.max_epochs = max_epochs
        self.epoch = 0
        self.version = version
        self.nickname = nickname
        self.model_name = model_name
        self.architecture = architecture
        self.start_time = None
        self.batches_processed = 0
        self.total_batches = len(self.train_dl)
        self.stop_event = threading.Event()
        self.accumulation_steps = HYPERPARAMS["gradient_accumulation_steps"]
        self.criterion = nn.L1Loss()
        self.scaler = torch.amp.GradScaler('cuda')
        self.best_loss = float('inf')
        self.patience = 5  
        self.patience_counter = 0
        self.scheduler = None
        self.text_area = text_area 
        self.warmup_epochs = warmup_epochs

    def adjust_learning_rate_warmup(self, epoch):
        if epoch < self.warmup_epochs:
            lr = HYPERPARAMS["initial_lr"] * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def run(self):
        self.start_time = time.time()
        self.net_G.to(self.device)
        self.net_G.train()

        if self.warmup_epochs > 0:
            self.text_area.append(f"Starting WarmUp {self.warmup_epochs} Epochs with learning rate: {HYPERPARAMS['initial_lr'] / self.warmup_epochs:.7f}")
        else:
            self.text_area.append("Skipping WarmUp phase as warmup_epochs is set to 0.")

        self.optimizer = optim.Adam(self.net_G.parameters(), lr=HYPERPARAMS["initial_lr"], betas=HYPERPARAMS["optimizer_betas"], weight_decay=HYPERPARAMS["weight_decay"])

        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=0
        )

        for self.epoch in range(self.max_epochs):
            if self.stop_event.is_set():
                break

            if self.warmup_epochs > 0 and self.epoch < self.warmup_epochs:
                self.adjust_learning_rate_warmup(self.epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Warmup Epoch {self.epoch + 1}, Learning Rate: {current_lr}")
            else:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Cosine Annealing Epoch {self.epoch + 1}, Learning Rate: {current_lr}")

            loss_meter = AverageMeter()
            l2_reg_meter = AverageMeter()
            epoch_start_time = time.time()
            self.batches_processed = 0

            self.train_dl.dataset.paths = np.random.permutation(self.train_dl.dataset.paths)

            self.optimizer.zero_grad()

            for i, data in enumerate(self.train_dl):
                if self.stop_event.is_set():
                    break

                iteration_start_time = time.time()

                L = data['L'].to(self.device)
                ab = data['ab'].to(self.device)

                with torch.amp.autocast('cuda'):
                    preds = self.net_G(L)
                    loss = self.criterion(preds, ab)
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                loss_meter.update(loss.item(), L.size(0))
                l2_reg = self.calculate_l2_regularization()
                l2_reg_meter.update(l2_reg, L.size(0))

                self.batches_processed += 1

                elapsed_time = time.time() - self.start_time
                elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
                epoch_elapsed_time = time.time() - epoch_start_time
                epoch_elapsed_time_str = str(timedelta(seconds=int(epoch_elapsed_time)))
                iteration_time = time.time() - iteration_start_time
                it_s = 1.0 / iteration_time if iteration_time > 0 else 0.0
                it_s_str = f"{it_s:.1f} it/s"
                estimated_epoch_time = epoch_elapsed_time * (len(self.train_dl) / (i + 1))
                estimated_epoch_time_str = str(timedelta(seconds=int(estimated_epoch_time)))
                estimated_total_time = estimated_epoch_time * (self.max_epochs - self.epoch)
                estimated_total_time_str = str(timedelta(seconds=int(estimated_total_time)))

                self.update_progress.emit(
                    int((i + 1) / len(self.train_dl) * 100),
                    f"Generator Training Epoch {self.epoch + 1}, Batch {i + 1}, L1 Loss: {loss_meter.avg:.4f} L2: {l2_reg_meter.avg:.4f}",
                    elapsed_time_str,
                    epoch_elapsed_time_str,
                    estimated_epoch_time_str,
                    estimated_total_time_str,
                    it_s_str
                )

            if loss_meter.avg < self.best_loss:
                self.best_loss = loss_meter.avg
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(loss_meter.avg)
            new_lr = self.optimizer.param_groups[0]['lr']

            if new_lr < old_lr:
                lr_message = f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}"
                self.text_area.append(lr_message)

            if self.patience_counter >= self.patience:
                self.text_area.append(f"Early stopping after {self.patience} epochs without improvement.")
                break

            epoch_summary = f"=== Generator Training Epoch {self.epoch + 1} Summary ===\n"
            epoch_summary += f"- Training Loss: {loss_meter.avg:.4f}\n"
            epoch_summary += f"- L2 Regularization: {l2_reg_meter.avg:.4f}\n"
            epoch_summary += f"- Learning Rate: {current_lr:.16f}\n"
            epoch_summary += f"- Elapsed Time: {elapsed_time_str}\n"
            epoch_summary += f"- Elapsed Epoch Time: {epoch_elapsed_time_str}\n"
            epoch_summary += f"- Estimated Time Remaining: {estimated_total_time_str}\n"
        
            logger.info(epoch_summary)
            self.update_progress.emit(100, epoch_summary, elapsed_time_str, epoch_elapsed_time_str, estimated_epoch_time_str, estimated_total_time_str, it_s_str)

        self.generator_training_finished.emit(self.net_G)

    def stop(self):
        self.stop_event.set()

    def calculate_l2_regularization(self):
        l2_reg = 0.0
        for param in self.net_G.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        return l2_reg.item()

class GANTrainingThread(QThread):
    update_progress = pyqtSignal(int, str, str, str, str, str, str)
    gan_training_finished = pyqtSignal(str)

    def __init__(
        self, 
        model, 
        train_dl, 
        val_dl, 
        device, 
        optimizer_G, 
        optimizer_D, 
        criterion_GAN, 
        criterion_L1, 
        save_dir, 
        max_epochs, 
        version, 
        nickname, 
        model_name, 
        architecture, 
        version_major, 
        version_letter, 
        infobox1
    ):
        super().__init__()
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.criterion_GAN = criterion_GAN
        self.criterion_L1 = criterion_L1
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.epoch = 0
        self.min_loss = float('inf')
        self.version = version
        self.nickname = nickname
        self.model_name = model_name
        self.architecture = architecture
        self.version_major = version_major
        self.version_letter = version_letter
        self.start_time = None
        self.batches_processed = 0
        self.total_batches = len(self.train_dl)
        self.initial_weights = [p.data.cpu().numpy() for p in self.model.net_G.parameters() if p.requires_grad]
        self.stop_event = threading.Event()
        self.accumulation_steps = HYPERPARAMS["gradient_accumulation_steps"]
        self.scaler = torch.amp.GradScaler('cuda')
        self.infobox1 = infobox1

    def calculate_l2_regularization(self):
        l2_reg = 0.0
        for param in self.model.net_G.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        return l2_reg.item()

    def save_model(self, epoch):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            self.save_dir,
            f"{self.model.architecture}_{self.model_name}_{self.nickname}_v{self.version}_epoch{epoch}_{timestamp}.pth"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.net_G.state_dict(),
            'discriminator_state_dict': self.model.net_G.discriminator.state_dict(),
            'optimizer_G_state_dict': self.model.opt_G.state_dict(),
            'optimizer_D_state_dict': self.model.opt_D.state_dict(),
            'loss': self.min_loss,
            'nickname': self.nickname,
            'model_name': self.model_name,
            'version': self.version, 
            'architecture': self.model.architecture,
        }, model_path)

        self.infobox1.append(f"Model saved as {model_path}")

    def run(self):
        self.start_time = time.time()
        self.model.to(self.device)
        self.model.train()

        for self.epoch in range(self.max_epochs):
            if self.stop_event.is_set():
                break

            loss_meter_dict = self.create_loss_meters()
            l2_reg_meter = AverageMeter()

            epoch_loss = 0
            epoch_start_time = time.time()
            self.batches_processed = 0

            self.train_dl.dataset.paths = np.random.permutation(self.train_dl.dataset.paths)

            for i, data in enumerate(self.train_dl):
                if self.stop_event.is_set():
                    break

                iteration_start_time = time.time()

                self.model.setup_input(data)
                self.model.forward()

                self.optimizer_D.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss_D, loss_D_fake, loss_D_real = self.model.backward_D()
                    loss_D = loss_D / self.accumulation_steps
                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.optimizer_D)
                self.scaler.update()

                self.optimizer_G.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss_G, loss_G_GAN, loss_G_L1 = self.model.backward_G()
                    loss_G = loss_G / self.accumulation_steps
                self.scaler.scale(loss_G).backward()
                self.scaler.step(self.optimizer_G)
                self.scaler.update()

                epoch_loss += loss_G.item()

                l2_reg = self.calculate_l2_regularization()
                l2_reg_meter.update(l2_reg, data['L'].size(0))

                self.update_losses(loss_meter_dict, loss_G.item(), loss_G_GAN.item(), loss_G_L1.item(), loss_D.item(), loss_D_fake.item(), loss_D_real.item())

                self.batches_processed += 1

                elapsed_time = time.time() - self.start_time
                elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
                epoch_elapsed_time = time.time() - epoch_start_time
                epoch_elapsed_time_str = str(timedelta(seconds=int(epoch_elapsed_time)))
                iteration_time = time.time() - iteration_start_time
                it_s = 1.0 / iteration_time if iteration_time > 0 else 0.0
                it_s_str = f"{it_s:.1f} it/s"
                estimated_epoch_time = epoch_elapsed_time * (len(self.train_dl) / (i + 1))
                estimated_epoch_time_str = str(timedelta(seconds=int(estimated_epoch_time)))
                estimated_total_time = estimated_epoch_time * (self.max_epochs - self.epoch)
                estimated_total_time_str = str(timedelta(seconds=int(estimated_total_time)))

                self.update_progress.emit(
                    int((i + 1) / len(self.train_dl) * 100),
                    f"GAN Training Epoch {self.epoch + 1}, Batch {i + 1}, Loss: {loss_G.item():.4f} (GAN: {loss_G_GAN.item():.4f}, L1: {loss_G_L1.item():.4f}, L2: {l2_reg_meter.avg:.4f})",
                    elapsed_time_str,
                    epoch_elapsed_time_str,
                    estimated_epoch_time_str,
                    estimated_total_time_str,
                    it_s_str
                )

            val_loss = self.validate()
            
            epoch_summary = f"=== GAN Training Epoch {self.epoch + 1} Summary ===\n"
            epoch_summary += f"- Training Loss: {epoch_loss / len(self.train_dl):.4f}\n"
            epoch_summary += f"- Validation Loss: {val_loss:.4f}\n"
            epoch_summary += f"- L2 Regularization: {l2_reg_meter.avg:.4f}\n"
            epoch_summary += f"- Elapsed Time: {elapsed_time_str}\n"
            epoch_summary += f"- Elapsed Epoch Time: {epoch_elapsed_time_str}\n"
            epoch_summary += f"- Estimated Time Remaining: {estimated_total_time_str}\n"
            epoch_summary += f"- Loss_G: {loss_G.item():.4f}\n"
            epoch_summary += f"- Loss_G_GAN: {loss_G_GAN.item():.4f}\n"
            epoch_summary += f"- Loss_G_L1: {loss_G_L1.item():.4f}\n"
            epoch_summary += f"- Loss_D: {loss_D.item():.4f}\n"
            epoch_summary += f"- Loss_D_fake: {loss_D_fake.item():.4f}\n"
            epoch_summary += f"- Loss_D_real: {loss_D_real.item():.4f}\n"
            
            logger.info(epoch_summary)
            self.update_progress.emit(100, epoch_summary, elapsed_time_str, epoch_elapsed_time_str, estimated_epoch_time_str, estimated_total_time_str, it_s_str)

            if (self.epoch + 1) % 5 == 0:
                self.save_model(self.epoch + 1)

        self.gan_training_finished.emit("") 

    def validate(self):
        self.model.eval()
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            for data in self.val_dl:
                self.model.setup_input(data)
                self.model.forward()
                loss_G, loss_G_GAN, loss_G_L1 = self.model.backward_G()
                val_loss_meter.update(loss_G.item())

        self.model.train()
        return val_loss_meter.avg

    def stop(self):
        self.stop_event.set()

    def create_loss_meters(self):
        loss_D_fake = AverageMeter()
        loss_D_real = AverageMeter()
        loss_D = AverageMeter()
        loss_G_GAN = AverageMeter()
        loss_G_L1 = AverageMeter()
        loss_G = AverageMeter()

        return {
            'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G
        }

    def update_losses(self, loss_meter_dict, loss_G, loss_G_GAN, loss_G_L1, loss_D, loss_D_fake, loss_D_real):
        loss_meter_dict['loss_G'].update(loss_G)
        loss_meter_dict['loss_G_GAN'].update(loss_G_GAN)
        loss_meter_dict['loss_G_L1'].update(loss_G_L1)
        loss_meter_dict['loss_D'].update(loss_D)
        loss_meter_dict['loss_D_fake'].update(loss_D_fake)
        loss_meter_dict['loss_D_real'].update(loss_D_real)

class ColorizationThread(QThread):
    update_progress = pyqtSignal(int, str, str, str, str, str)
    colorization_finished = pyqtSignal()

    def __init__(self, model, folder_path, device):
        super().__init__()
        self.model = model
        self.folder_path = folder_path
        self.device = device
        self.start_time = time.time()

    def run(self):
        self.model.to(self.device)
        self.model.eval()

        output_folder = os.path.join(self.folder_path, HYPERPARAMS["colorized_folder_name"])
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"Colorizing images in: {self.folder_path}")

        image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_images = len(image_files)
        processed_images = 0

        for img_name in image_files:
            img_path = os.path.join(self.folder_path, img_name)
            img = Image.open(img_path).convert('RGB')
            original_size = img.size

            img_lab = rgb2lab(np.array(img)).astype(np.float32)
            img_l = img_lab[:, :, 0] / 100.0
            img_l = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    ab_output = self.model.net_G(img_l)

            L = (img_l.squeeze().cpu().numpy() * 100.0)
            ab = ab_output.squeeze().cpu().numpy() * 110.0

            Lab = np.stack([L, ab[0], ab[1]], axis=-1)
            xyz = lab2xyz(Lab)
            srgb = self.xyz_to_srgb(xyz)

            colorized_img = Image.fromarray(srgb)
            output_img_path = os.path.join(output_folder, "colorized_" + img_name)
            colorized_img.save(output_img_path)
            processed_images += 1

            progress = int((processed_images / total_images) * 100)
            elapsed_time = time.time() - self.start_time
            elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
            predicted_epoch_time = "00:00:00"
            predicted_total_time = "00:00:00"
            it_s = f"{(processed_images / elapsed_time):.1f} it/s" if elapsed_time > 0 else "0.0 it/s"

            self.update_progress.emit(
                progress,
                f"Processed {processed_images}/{total_images} images",
                elapsed_time_str,
                predicted_epoch_time,
                predicted_total_time,
                it_s
            )

            logger.info(f"Saved colorized image: {output_img_path} ({processed_images}/{total_images})")

            clear_gpu_memory()

        logger.info("Colorization completed.")
        self.colorization_finished.emit()

    def xyz_to_srgb(self, xyz):
        xyz_to_rgb_matrix = np.array([
            [3.2406255, -1.537208, -0.4986286],
            [-0.9689307, 1.8757561, 0.0415175],
            [0.0557101, -0.2040211, 1.0569959]
        ])

        linear_rgb = np.dot(xyz, xyz_to_rgb_matrix.T)
        linear_rgb = np.clip(linear_rgb, 0, 1)

        def transfer(c):
            return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)

        srgb = transfer(linear_rgb)
        return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Supercolor by nitrocon")
        self.setGeometry(100, 100, HYPERPARAMS["app_width"], HYPERPARAMS["app_height"])

        self.device, self.device_info = get_device()
        self.model = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.criterion_GAN = GANLoss()
        self.criterion_L1 = nn.L1Loss()

        self.version_major = 0
        self.version_minor = 0
        self.version_letter = 'a'
        self.version = f"{self.version_major:02d}{self.version_letter}"
        self.nickname = HYPERPARAMS["nickname"]
        self.model_name = HYPERPARAMS["model_name"]

        self.current_theme = "dark"
        self.initUI()
        self.apply_theme(self.current_theme)

    def apply_theme(self, theme_name):
        self.current_theme = theme_name
        self.setStyleSheet(THEMES[theme_name])

    def initUI(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        gui1_panel = QWidget()
        gui1_layout = QVBoxLayout()

        button_layout = QVBoxLayout()

        self.architecture_dropdown = QComboBox(self)
        self.architecture_dropdown.addItem("Dynamic UNET")
        self.architecture_dropdown.model().item(0).setEnabled(False)
        self.architecture_dropdown.model().item(0).setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.architecture_dropdown.insertSeparator(self.architecture_dropdown.count())
        self.architecture_dropdown.addItem("EfficientNet_B0", "efficientnetb0")
        self.architecture_dropdown.addItem("ResNet18", "resnet18")
        self.architecture_dropdown.addItem("ResNet34", "resnet34")
        self.architecture_dropdown.addItem("ResNet50", "resnet50")
        self.architecture_dropdown.addItem("ShuffleNetV2_X0.5", "shufflenetv2x0_5")
        self.architecture_dropdown.addItem("ShuffleNetV2_X1.0", "shufflenetv2x1_0")
        self.architecture_dropdown.addItem("ShuffleNetV2_X1.5", "shufflenetv2x1_5")
        self.architecture_dropdown.addItem("ShuffleNetV2_X2.0", "shufflenetv2x2_0")
        self.architecture_dropdown.insertSeparator(self.architecture_dropdown.count())

        button_layout.addWidget(QLabel("Select Architecture:"))
        button_layout.addWidget(self.architecture_dropdown)

        self.select_button = QPushButton('Select Folder', self)
        self.select_button.clicked.connect(self.select_folder)
        button_layout.addWidget(self.select_button)

        self.nickname_label = QLabel("Enter Nickname:", self)
        self.nickname_input = QLineEdit(self)
        self.nickname_input.setText(self.nickname)
        button_layout.addWidget(self.nickname_label)
        button_layout.addWidget(self.nickname_input)

        self.model_name_label = QLabel("Enter Model Name:", self)
        self.model_name_input = QLineEdit(self)
        self.model_name_input.setText(self.model_name)
        button_layout.addWidget(self.model_name_label)
        button_layout.addWidget(self.model_name_input)

        self.start_button = QPushButton('Start Training', self)
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Training', self)
        self.stop_button.clicked.connect(self.stop_training)
        button_layout.addWidget(self.stop_button)

        self.load_model_button = QPushButton('Load Model', self)
        self.load_model_button.clicked.connect(self.load_model)
        button_layout.addWidget(self.load_model_button)

        self.new_model_button = QPushButton('New Model', self)
        self.new_model_button.clicked.connect(self.new_model)
        button_layout.addWidget(self.new_model_button)

        self.colorize_button = QPushButton('Colorize Images in Folder', self)
        self.colorize_button.clicked.connect(self.colorize_images_in_folder)
        button_layout.addWidget(self.colorize_button)

        self.elapsed_time_label = QLabel("Elapsed Time: 00:00:00", self)
        button_layout.addWidget(self.elapsed_time_label)

        self.elapsed_epoch_time_label = QLabel("Elapsed Epoch Time: 00:00:00", self)
        button_layout.addWidget(self.elapsed_epoch_time_label)

        self.predicted_total_time_label = QLabel("Predicted Time: 00:00:00", self)
        button_layout.addWidget(self.predicted_total_time_label)

        self.predicted_epoch_time_label = QLabel("Predicted Epoch Time: 00:00:00", self)
        button_layout.addWidget(self.predicted_epoch_time_label)

        self.it_s_label = QLabel("Iterations/s: 0.0 it/s", self)
        button_layout.addWidget(self.it_s_label)

        self.status_label = QLabel("Select Folder to Start")
        self.status_label.setWordWrap(True)
        button_layout.addWidget(self.status_label)
        self.status_label.setFixedHeight(30)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        button_layout.addWidget(self.progress_bar)

        spacer = QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        button_layout.addSpacerItem(spacer)

        gui1_layout.addLayout(button_layout)

        gui1_panel.setFixedHeight(HYPERPARAMS["app_height"])
        gui1_panel.setLayout(gui1_layout)
        gui1_panel.setFixedWidth(HYPERPARAMS["gui1_width"])

        gui2_panel = QWidget()
        gui2_layout = QVBoxLayout()

        epochs_group = QGroupBox("Epochs - Training")
        epochs_layout = QHBoxLayout()
        epochs_layout.setSpacing(HYPERPARAMS["widget_spacing"])

        self.warmup_epochs_label = QLabel("Warm Up:", self)
        self.warmup_epochs_input = QLineEdit(self)
        self.warmup_epochs_input.setText(str(HYPERPARAMS["warmup_epochs"]))
        epochs_layout.addWidget(self.warmup_epochs_label)
        epochs_layout.addWidget(self.warmup_epochs_input)

        self.epoch_input_label = QLabel("Gen.:", self)
        self.epoch_input = QLineEdit(self)
        self.epoch_input.setText(str(HYPERPARAMS["generator_train_epochs"]))
        epochs_layout.addWidget(self.epoch_input_label)
        epochs_layout.addWidget(self.epoch_input)

        self.gan_epoch_input_label = QLabel("GAN:", self)
        self.gan_epoch_input = QLineEdit(self)
        self.gan_epoch_input.setText(str(HYPERPARAMS["gan_train_epochs"]))
        epochs_layout.addWidget(self.gan_epoch_input_label)
        epochs_layout.addWidget(self.gan_epoch_input)

        epochs_group.setLayout(epochs_layout)
        epochs_group.setFixedHeight(HYPERPARAMS["epochs_groupbox_height"])
        gui2_layout.addWidget(epochs_group)

        batch_size_layout = QHBoxLayout()
        batch_size_layout.setSpacing(HYPERPARAMS["widget_spacing"])

        self.batch_size_label = QLabel("Batch Size:", self)
        self.batch_size_input = QLineEdit(self)
        self.batch_size_input.setText(str(HYPERPARAMS["batch_size"]))
        self.batch_size_input.textChanged.connect(self.update_dataloaders)
        batch_size_layout.addWidget(self.batch_size_label)
        batch_size_layout.addWidget(self.batch_size_input)

        self.image_size_label = QLabel("Image Size:", self)
        self.image_size_input = QLineEdit(self)
        self.image_size_input.setText(str(HYPERPARAMS["image_size"]))
        self.image_size_input.textChanged.connect(self.update_image_size)
        batch_size_layout.addWidget(self.image_size_label)
        batch_size_layout.addWidget(self.image_size_input)

        batch_size_widget = QWidget()
        batch_size_widget.setLayout(batch_size_layout)
        batch_size_widget.setFixedHeight(HYPERPARAMS["widget_height"])
        gui2_layout.addWidget(batch_size_widget)

        theme_layout = QHBoxLayout()
        theme_layout.setSpacing(HYPERPARAMS["widget_spacing"])

        self.theme_label = QLabel("Select Theme:", self)
        self.theme_dropdown = QComboBox(self)
        self.theme_dropdown.addItem("Dark")
        self.theme_dropdown.addItem("Light")
        self.theme_dropdown.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(self.theme_label)
        theme_layout.addWidget(self.theme_dropdown)

        theme_widget = QWidget()
        theme_widget.setLayout(theme_layout)
        theme_widget.setFixedHeight(HYPERPARAMS["widget_height"])
        gui2_layout.addWidget(theme_widget)

        for i in range(4, 11):
            empty_widget = QWidget()
            empty_widget.setFixedHeight(HYPERPARAMS["widget_height"])
            gui2_layout.addWidget(empty_widget)

        gui2_panel.setLayout(gui2_layout)
        gui2_panel.setFixedWidth(HYPERPARAMS["gui2_width"])

        self.infobox1 = QTextEdit(self)
        self.infobox1.setReadOnly(True)
        self.infobox1.setFixedWidth(HYPERPARAMS["infobox1_width"])

        self.infobox2 = QTextEdit(self)
        self.infobox2.setReadOnly(True)
        self.infobox2.setFixedWidth(HYPERPARAMS["app_width"] - HYPERPARAMS["gui1_width"] - HYPERPARAMS["gui2_width"] - HYPERPARAMS["infobox1_width"])

        splitter.addWidget(gui1_panel)
        splitter.addWidget(gui2_panel)
        splitter.addWidget(self.infobox1)
        splitter.addWidget(self.infobox2)

        self.setCentralWidget(splitter)

        self.infobox1.append(self.device_info)

    def change_theme(self, theme_name):
        self.apply_theme(theme_name.lower())

    def update_progress(self, value, message, elapsed_time="00:00:00", elapsed_epoch_time="00:00:00", predicted_epoch_time="00:00:00", predicted_total_time="00:00:00", it_s="0.0 it/s"):
        self.progress_bar.setValue(value)
        self.elapsed_time_label.setText(f"Elapsed Time: {elapsed_time}")
        self.predicted_total_time_label.setText(f"Predicted Total Time: {predicted_total_time}")
        self.elapsed_epoch_time_label.setText(f"Elapsed Epoch Time: {elapsed_epoch_time}")
        self.predicted_epoch_time_label.setText(f"Predicted Epoch Time: {predicted_epoch_time}")
        self.it_s_label.setText(f"Iterations/s: {it_s}")

        if "Summary" in message:
            self.infobox1.append(message)

    def update_infobox2(self, value, message, elapsed_time="00:00:00", elapsed_epoch_time="00:00:00", predicted_epoch_time="00:00:00", predicted_total_time="00:00:00", it_s="0.0 it/s"):
        if "Summary" not in message:
            self.infobox2.append(message)

    def update_image_size(self):
        try:
            image_size = int(self.image_size_input.text())
            if image_size > 0:
                HYPERPARAMS["image_size"] = image_size
                self.infobox1.append(f"Image size updated to: {image_size}")
            else:
                self.infobox1.append("Image size must be greater than 0.")
        except ValueError:
            self.infobox1.append("Invalid input for image size. Please enter a valid integer.")

    def update_dataloaders(self):
        if hasattr(self, 'train_paths') and hasattr(self, 'val_paths'):
            batch_size = int(self.batch_size_input.text())
            image_size = int(self.image_size_input.text())
            HYPERPARAMS["image_size"] = image_size
            self.train_dl = make_dataloaders(paths=self.train_paths, split='train', batch_size=batch_size)
            self.val_dl = make_dataloaders(paths=self.val_paths, split='val', batch_size=batch_size)
            self.infobox1.append(f"DataLoader updated with batch size: {batch_size} and image size: {image_size}")

    def select_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.folder_path:
            self.data_preparation_thread = DataPreparationThread(self.folder_path)
            self.data_preparation_thread.data_prepared.connect(self.on_data_prepared)
            self.data_preparation_thread.start()

    def on_data_prepared(self, train_paths, val_paths):
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.update_dataloaders()

        max_length = HYPERPARAMS["max_image_length"]
        if len(self.folder_path) > max_length:
            display_path = self.folder_path[:max_length] + "..." + self.folder_path[-max_length:]
        else:
            display_path = self.folder_path
        self.status_label.setText(f"Selected Folder: {display_path}")
        self.status_label.setToolTip(self.folder_path)

        dataset_info = (
            f"=== Dataset Information ===\n"
            f"Selected Folder: {display_path}\n"
            f"Training Images: {len(self.train_paths)}\n"
            f"Validation Images: {len(self.val_paths)}\n"
            f"Batch Size: {self.batch_size_input.text()}\n"
            f"Image Size: {self.image_size_input.text()}\n"
        )
        self.infobox1.append(dataset_info)

    def start_training(self):
        if not hasattr(self, 'train_dl') or self.train_dl is None:
            self.infobox1.append("No dataset selected. Please select a folder first.")
            return

        self.update_dataloaders()

        selected_architecture = self.architecture_dropdown.currentData()
        if self.model is None or selected_architecture != self.architecture_dropdown.currentData():
            if selected_architecture == "resnet18":
                from architectures.dynamicunet.resnet18 import ResNet18DynamicUnetColorizationModel
                net_G = ResNet18DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"])
            elif selected_architecture == "resnet50":
                from architectures.dynamicunet.resnet50 import ResNet50DynamicUnetColorizationModel
                net_G = ResNet50DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"])
            elif selected_architecture == "efficientnetb0":
                from architectures.dynamicunet.efficientnetb0 import EfficientNetB0DynamicUnet
                net_G = EfficientNetB0DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif selected_architecture == "resnet34":
                from architectures.dynamicunet.resnet34 import ResNet34DynamicUnetColorizationModel
                net_G = ResNet34DynamicUnetColorizationModel(self.device, size=HYPERPARAMS["image_size"])
            elif selected_architecture == "shufflenetv2x0_5":
                from architectures.dynamicunet.shufflenetv2x05 import ShuffleNetV2x05DynamicUnet
                net_G = ShuffleNetV2x05DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif selected_architecture == "shufflenetv2x1_0":
                from architectures.dynamicunet.shufflenetv2x1_0 import ShuffleNetV2x1_0DynamicUnet
                net_G = ShuffleNetV2x1_0DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif selected_architecture == "shufflenetv2x1_5":
                from architectures.dynamicunet.shufflenetv2x1_5 import ShuffleNetV2x1_5DynamicUnet
                net_G = ShuffleNetV2x1_5DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            elif selected_architecture == "shufflenetv2x2_0":
                from architectures.dynamicunet.shufflenetv2x2_0 import ShuffleNetV2x2_0DynamicUnet
                net_G = ShuffleNetV2x2_0DynamicUnet(self.device, size=HYPERPARAMS["image_size"])
            else:
                self.infobox1.append("Invalid architecture selected.")
                return

            self.model = MainModel(net_G=net_G, architecture=selected_architecture)
            self.infobox1.append(f"New model created with architecture: {selected_architecture}")
        else:
            self.infobox1.append(f"Continue loaded model: {self.model_name}")

        self.infobox1.append(f"Using L2 regularization (weight_decay): {HYPERPARAMS['weight_decay']}")

        self.status_label.setText("Generator Training Started...")
        self.progress_bar.setValue(0)

        max_epochs_generator = int(self.epoch_input.text())
        max_epochs_gan = int(self.gan_epoch_input.text())

        self.nickname = self.nickname_input.text()
        self.model_name = self.model_name_input.text()

        warmup_epochs = int(self.warmup_epochs_input.text())

        if warmup_epochs == 0 and max_epochs_generator == 0:
            self.infobox1.append("Skipping Generator Training. Starting GAN training directly...")
            self.start_gan_training()
        else:
            self.generator_training_thread = GeneratorTrainingThread(
                net_G=self.model.net_G,
                train_dl=self.train_dl,
                device=self.device,
                max_epochs=max_epochs_generator,
                version=self.version,
                nickname=self.nickname,
                model_name=self.model_name,
                architecture=selected_architecture,
                text_area=self.infobox1,
                warmup_epochs=warmup_epochs
            )
            self.generator_training_thread.update_progress.connect(self.update_progress)
            self.generator_training_thread.update_progress.connect(self.update_infobox2)
            self.generator_training_thread.generator_training_finished.connect(self.on_generator_training_finished)
            self.generator_training_thread.start()

    def start_gan_training(self):
        self.infobox1.append("Starting GAN training...")
        self.status_label.setText("GAN Training Started...")

        self.gan_training_thread = GANTrainingThread(
            self.model, 
            self.train_dl, 
            self.val_dl, 
            self.device, 
            self.model.opt_G, 
            self.model.opt_D, 
            self.model.GANcriterion, 
            self.model.L1criterion, 
            "models", 
            int(self.gan_epoch_input.text()),
            self.version, 
            self.nickname, 
            self.model_name, 
            self.architecture_dropdown.currentData(),
            self.version_major, 
            self.version_letter, 
            self.infobox1
        )

        self.gan_training_thread.update_progress.connect(self.update_progress)
        self.gan_training_thread.update_progress.connect(self.update_infobox2)
        self.gan_training_thread.gan_training_finished.connect(self.gan_training_finished)
        self.gan_training_thread.start()

    def on_generator_training_finished(self, net_G):
        self.infobox1.append("Generator training completed. Starting GAN training...")
        self.status_label.setText("GAN Training Started...")

        self.gan_training_thread = GANTrainingThread(
            self.model, 
            self.train_dl, 
            self.val_dl, 
            self.device, 
            self.model.opt_G, 
            self.model.opt_D, 
            self.model.GANcriterion, 
            self.model.L1criterion, 
            "models", 
            int(self.gan_epoch_input.text()),
            self.version, 
            self.nickname, 
            self.model_name, 
            self.architecture_dropdown.currentData(),
            self.version_major, 
            self.version_letter, 
            self.infobox1
        )

        self.gan_training_thread.update_progress.connect(self.update_progress)
        self.gan_training_thread.update_progress.connect(self.update_infobox2)
        self.gan_training_thread.gan_training_finished.connect(self.gan_training_finished)
        self.gan_training_thread.start()

    def stop_training(self):
        if hasattr(self, 'generator_training_thread'):
            self.generator_training_thread.stop()
            self.status_label.setText("Generator Training Stopped")
            self.infobox1.append("Generator training stopped by user.")
        if hasattr(self, 'gan_training_thread'):
            self.gan_training_thread.stop()
            self.status_label.setText("GAN Training Stopped")
            self.infobox1.append("GAN training stopped by user.")

    def display_model_info(self, epoch, loss):
        model_info = f"=== Model Information ===\n"
        model_info += f"Epoch: {epoch}\n"
        model_info += f"Loss: {loss:.4f}\n"
        model_info += f"Version: {self.version}\n"
        model_info += f"Nickname: {self.nickname}\n"
        model_info += f"Model Name: {self.model_name}\n"
        model_info += f"Model Architecture: {self.model.architecture}\n"
        self.infobox1.append(model_info)

    def gan_training_finished(self, model_path):
        self.status_label.setText("GAN Training Finished")
    
        self.version_major += 1  
        if self.version_major > 99:
            self.version_major = 0
            self.version_letter = chr(ord(self.version_letter) + 1)  
            if self.version_letter > 'z':
                self.version_letter = 'a' 
        self.version = f"{self.version_major:02d}{self.version_letter}" 

        self.model.version = self.version

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            "models",
            f"{self.model.architecture}_{self.model_name}_{self.nickname}_v{self.version}_{timestamp}.pth"
        )

        torch.save({
            'epoch': self.gan_training_thread.epoch,
            'model_state_dict': self.model.net_G.state_dict(),
            'discriminator_state_dict': self.model.net_G.discriminator.state_dict(),
            'optimizer_G_state_dict': self.model.opt_G.state_dict(),
            'optimizer_D_state_dict': self.model.opt_D.state_dict(),
            'loss': self.gan_training_thread.min_loss,
            'nickname': self.nickname,
            'model_name': self.model_name,
            'version': self.version, 
            'architecture': self.model.architecture,
        }, model_path)

        self.infobox1.append(f"Model saved as {model_path}")
        self.infobox1.append(f"Version updated to: {self.version}")

        torch.cuda.empty_cache()
        gc.collect()

        initial_weights = self.gan_training_thread.initial_weights
        final_weights = [p.data.cpu().numpy() for p in self.model.net_G.parameters() if p.requires_grad]
        weight_change = 0.0
        for initial, final in zip(initial_weights, final_weights):
            weight_change += np.sum(np.abs(final - initial))

        total_weights = sum(p.numel() for p in self.model.net_G.parameters() if p.requires_grad)
        weight_change_percent = (weight_change / total_weights) * 100

        self.infobox1.append(f"Weight change: {weight_change_percent:.2f}%")

        self.display_model_info(epoch=self.gan_training_thread.epoch, loss=self.gan_training_thread.min_loss)

    def load_model(self):
        torch.cuda.empty_cache()
        gc.collect()

        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth)")
        if self.file_path:
            self.model_loading_thread = ModelLoadingThread(self.file_path, self.device)
            self.model_loading_thread.model_loaded.connect(self.on_model_loaded)
            self.model_loading_thread.start()

    def on_model_loaded(self, model):
        if model is None:
            self.infobox1.append("Failed to load model.")
            return

        self.model = model
        self.infobox1.append("Model loaded successfully.")

        self.nickname = self.model.nickname
        self.model_name = self.model.model_name
        self.version = self.model.version

        if len(self.version) == 3:
            self.version_major = int(self.version[:2])
            self.version_letter = self.version[2]
        else:
            self.infobox1.append("Invalid version format in loaded model.")
            self.version_major = 0
            self.version_letter = 'a'

        self.nickname_input.setText(self.nickname)
        self.model_name_input.setText(self.model_name)

        architecture = self.model.architecture if hasattr(self.model, 'architecture') else 'resnet18'
        index = self.architecture_dropdown.findData(architecture)
        if index >= 0:
            self.architecture_dropdown.setCurrentIndex(index)
        else:
            self.infobox1.append(f"Architecture {architecture} not found in dropdown.")

        num_layers = sum(1 for _ in self.model.net_G.children())
        num_parameters = sum(p.numel() for p in self.model.net_G.parameters())
        num_weights = sum(p.numel() for p in self.model.net_G.parameters() if p.requires_grad)

        model_info = f"=== Model Information ===\n"
        model_info += f"Model loaded from: {self.file_path}\n"
        model_info += f"Model Name: {self.model_name}\n"
        model_info += f"Model Architecture: {architecture}\n"
        model_info += f"Number of layers: {num_layers}\n"
        model_info += f"Number of parameters: {num_parameters:,}\n"
        model_info += f"Number of weights: {num_weights:,}\n"
        model_info += f"Nickname: {self.nickname}\n"
        model_info += f"Version: {self.version}\n"

        self.infobox1.append(model_info)

    def new_model(self):
        self.model = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.version_major = 0
        self.version_minor = 0
        self.version_letter = 'a'
        self.version = f"{self.version_major:02d}{self.version_letter}"
        self.nickname = HYPERPARAMS["nickname"]
        self.model_name = HYPERPARAMS["model_name"]
        self.nickname_input.setText(self.nickname)
        self.model_name_input.setText(self.model_name)
        self.infobox1.append("=== New Model Created ===\nModel reset. Ready to create a new model.")

    def colorize_images_in_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(None, "Select Folder with Grayscale Images")
        if self.folder_path:
            self.colorization_thread = ColorizationThread(self.model, self.folder_path, self.device)
            self.colorization_thread.update_progress.connect(self.update_infobox2)
            self.colorization_thread.colorization_finished.connect(self.colorization_finished)
            self.colorization_thread.start()

    def colorization_finished(self):
        self.infobox1.append("=== Colorization Status ===\nColorization completed.")
        self.status_label.setText("Colorization Finished")
        clear_gpu_memory()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
