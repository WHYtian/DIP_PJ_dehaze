import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
from model import DM2FNet, DM2FNet_woPhy
from torch.utils.data import Dataset,DataLoader
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import deltaE_ciede2000, rgb2lab
import datetime
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

device = torch.device("cuda")
to_tensor = ToTensor()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

to_pil = transforms.ToPILImage()

net = DM2FNet().to(device)

model_path = '/mnt/RESIDE_ITS/nochange/iter_40000_loss_0.01183_lr_0.000000.pth'

net.load_state_dict(torch.load(model_path))

filenames = [f for f in os.listdir('/root/DM2F-Net-master/pic')]
for file in filenames:
    image_path = '/root/DM2F-Net-master/pic/'+file

    haze = Image.open(image_path).convert('RGB')
    haze = to_tensor(haze).unsqueeze(0)
    haze = haze.to(device)

    res = net(haze)

    r = res[0].detach().cpu().numpy().transpose([1, 2, 0])

    dehaze_path = '/root/DM2F-Net-master/pic/nochangeres/'+file

    to_pil(r).save(dehaze_path)