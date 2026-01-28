import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
from skimage.exposure import rescale_intensity
import cv2
import argparse
from model.FWMNet import FWMNet

parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='./datasets/traindata/train/input/raw', type=str, help='Input images')
parser.add_argument('--result_dir', default='./results/1030', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./checkpoints/FWMNet-64-4-dwt1-patch512/models/model_bestSSIM.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.PNG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.TIF'))
                  + glob(os.path.join(inp_dir, '*.tif')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights
model = FWMNet(in_chn=3, wf=64, depth=4)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')

mul = 16
index = 0
psnr_val_rgb = []
input_ = torch.zeros(1,3,1024,1024).cuda()
for file_ in files:
    # img = Image.open(file_).convert('RGB')
    # input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    img = Image.open(file_) # 适配单通道 TIF
    input_tmp = (TF.to_tensor(img).unsqueeze(0)/65536).cuda()
    input_[0,0,:,:] = input_tmp[0,0,:,:]
    input_[0,1,:,:] = input_[0,0,:,:]
    input_[0,2,:,:] = input_[0,0,:,:]
    # Pad the input if not_multiple_of 8
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    with torch.no_grad():
        restored = model(input_)

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored_normalized = rescale_intensity(restored, in_range=(0, 1), out_range=(0, 1))
    restored = img_as_ubyte(restored_normalized[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)
    index += 1
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')
