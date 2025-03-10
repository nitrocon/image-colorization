import torch
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet
from tqdm import tqdm

def build_res_unet(n_input=1, n_output=2, size=None):
    body = create_body(resnet34(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G

class ResNet34DynamicUnetColorizationModel(torch.nn.Module):
    def __init__(self, device, size=None):
        super(ResNet34DynamicUnetColorizationModel, self).__init__()
        self.device = device
        
        with tqdm(total=2, desc="Initializing ResNet34 Dynamic-UNet") as pbar:
            self.net_G = build_res_unet(n_input=1, n_output=2, size=size)
            pbar.update(1)
            
            self.net_G.to(self.device)
            pbar.update(1)
        
        print("ResNet34 Dynamic-UNet model initialized and moved to device.")

    def forward(self, x):
        x = x.to(self.device)
        return self.net_G(x)