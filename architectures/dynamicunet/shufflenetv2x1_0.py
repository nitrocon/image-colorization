import torch
from fastai.vision.learner import create_body
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from fastai.vision.models.unet import DynamicUnet
from tqdm import tqdm

def build_shufflenet_unet(n_input=1, n_output=2, size=None):
    shufflenet = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
    body = create_body(shufflenet, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G

class ShuffleNetV2x1_0DynamicUnet(torch.nn.Module):
    def __init__(self, device, size=None):
        super(ShuffleNetV2x1_0DynamicUnet, self).__init__()
        self.device = device
        
        with tqdm(total=2, desc="Initializing ShuffleNetV2 X1.0 Dynamic-UNet") as pbar:
            self.net_G = build_shufflenet_unet(n_input=1, n_output=2, size=size)
            pbar.update(1)
            
            self.net_G.to(self.device)
            pbar.update(1)
        
        print("ShuffleNetV2 X1.0 Dynamic-UNet model initialized and moved to device.")

    def forward(self, x):
        x = x.to(self.device)
        return self.net_G(x)