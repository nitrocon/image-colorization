import torch
from fastai.vision.learner import create_body
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from fastai.vision.models.unet import DynamicUnet
from tqdm import tqdm

def build_efficientnetb0_dynamicunet(n_input=1, n_output=2, size=None):
    efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    body = create_body(efficientnet, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G

class EfficientNetB0DynamicUnet(torch.nn.Module):
    def __init__(self, device, size=None):
        super(EfficientNetB0DynamicUnet, self).__init__()
        self.device = device
        
        with tqdm(total=2, desc="Initializing EfficientNetB0DynamicUnet") as pbar:
            self.net_G = build_efficientnetb0_dynamicunet(n_input=1, n_output=2, size=size)
            pbar.update(1)
            
            self.net_G.to(self.device)
            pbar.update(1)
        
        print("EfficientNetB0DynamicUnet model initialized and moved to device.")

    def forward(self, x):
        x = x.to(self.device)
        return self.net_G(x)