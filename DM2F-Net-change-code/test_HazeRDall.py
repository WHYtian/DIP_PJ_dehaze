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

ckpt_path = './ckpt'
#save_path = '/mnt/ckpt'
save_path = './ckpt'
exp_name = 'RESIDE_ITS'
#exp_name = 'O-Haze'



to_test = {
    #'O-Haze': 'data/HazeRD/data/simu',
    'O-Haze': '/mnt/data/HazeRD/data/simu',
}

to_pil = transforms.ToPILImage()
model_name = 'add_first'
class HazeRD(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        haze, gt, name = self.data[index]
        return haze, gt, name

    def __len__(self):
        return len(self.data)

def main(args):
    # log_path = os.path.join('/mnt/result_test', args['snapshot']+model_name+ '.txt')
    filenames = [file for file in os.listdir('/mnt/data/HazeRD/data/simu') if file.endswith('.jpg')]
    with torch.no_grad():
        criterion = nn.L1Loss().to(device)
        net = DM2FNet().to(device)
        #net = DM2FNet_woPhy().to(device)
        if len(args['snapshot']) > 0:
            print('load snapshot \'%s\' for testing' % args['snapshot'])
            #net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
            net.load_state_dict(torch.load(os.path.join('/mnt/RESIDE_ITS/nochange', args['snapshot'] + '.pth')))
        dataset = []
        net.eval()
        name = 'HazeRD'
        for file in filenames:
            temp = file.split('_')
            ori_name = temp[0]+'_'+temp[1]
            gt = Image.open(os.path.join('/mnt/data/HazeRD/data/img', ori_name+'_RGB.jpg')).convert('RGB')
            gt = to_tensor(gt)
            #print(gt)
            haze = Image.open(os.path.join('/mnt/data/HazeRD/data/simu', file)).convert('RGB')
            haze = to_tensor(haze)
            filename = file.split('.')[0]
            dataset.append((haze, gt, filename))

        dataset = HazeRD(dataset)
        # open(log_path, 'w+').write(name+ " "+ args['snapshot'] +model_name+ '\n\n')
        dataloader = DataLoader(dataset, batch_size=1)
        psnrs, ssims, ciede2000s = [], [], []
        loss_record = AvgMeter()
        
        for idx, data in enumerate(dataloader):
            # haze_image, _, _, _, fs = data
            haze, gts, fs = data
            # print(haze.shape, gts.shape)

            check_mkdir(os.path.join(save_path, exp_name,
                                        '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

            haze = haze.to(device)

            if 'O-Haze' in name:
                res = sliding_forward(net, haze).detach()
            else:
                #res = net(haze).detach()
                res = net(haze)
                #res = net(haze)
            # for l in [x_phy, x_j1, x_j2, x_j3, x_j4, x_j5, x_t]:
            #     nan_mask = torch.isnan(l.detach())
            #     nan_indices = torch.nonzero(nan_mask)
            #     print("NaN 值的位置下标:", nan_indices)

            loss = criterion(res.detach(), gts.to(device))
            loss_record.update(loss.item(), haze.size(0))

            for i in range(len(fs)):
                r = res[i].cpu().numpy().transpose([1, 2, 0])
                gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                labr = rgb2lab(r)
                labg = rgb2lab(gt)
                psnr = peak_signal_noise_ratio(gt, r)
                psnrs.append(psnr)
                ssim = structural_similarity(gt, r, win_size=3, data_range=1, multichannel=True,
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                ssims.append(ssim)
                #ciede2000 = deltaE_ciede2000((gt*255).astype(int), (r*255).astype(int)).mean()
                ciede2000 = deltaE_ciede2000(labg, labr).mean()
                ciede2000s.append(ciede2000)
                # print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}'
                #         .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, ciede2000))
                # log = 'predicting for %s (%d/%d) [%s]: PSNR %.4f, SSIM %.4f, CIEDE2000 %.4f' %\
                #         (name, idx + 1, len(dataloader), fs[i], psnr, ssim, ciede2000)
                # open(log_path, 'a').write(log + '\n')
            # for r, f in zip(res.cpu(), fs):
            #     dehaze_path = os.path.join(save_path, exp_name,
            #                         '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s_%s.jpg' % (f, t))

            #     to_pil(r).save(dehaze_path)
                
                # dehaze = Image.open(dehaze_path)
                # dehaze = to_tensor(dehaze).numpy().transpose([1, 2, 0])
                # # print(f'r{r.numpy().transpose([1, 2, 0])}')
                # # print(f'dehaze{dehaze}')
                # gt_path = '/mnt/data/HazeRD/data/img/'+f.split('_')[0]+'_'+f.split('_')[1]+'_RGB.jpg'
                # gt = Image.open(gt_path).convert('RGB')
                # gt = to_tensor(gt).numpy().transpose([1, 2, 0])
                # # print(gt_path)
                # ssim = structural_similarity(gt, dehaze, channel_axis=-1, data_range=1, multichannel=True,
                #             gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                # ciede2000 = deltaE_ciede2000(gt, dehaze).mean()
                # #print(f'{f}:SSIM {ssim}, CIEDE2000 {ciede2000}')
                # log = "%s: SSIM %.6f, CIEDE2000 %.6f" % (f, ssim, ciede2000)
                # open(log_path, 'a').write(log + '\n')
        print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}")
        # log = "[{name}] L1: %.6f, PSNR: %.6f, SSIM: %.6f, CIEDE2000: %.6f" %\
        #             (loss_record.avg, np.mean(psnrs), np.mean(ssims), np.mean(ciede2000s))
        # open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    argss = [{
    'snapshot': 'iter_40000_loss_0.01183_lr_0.000000',
},
{
    'snapshot': 'iter_35000_loss_0.01261_lr_0.000077',
},
{
    'snapshot': 'iter_30000_loss_0.01324_lr_0.000144',
},
{
    'snapshot': 'iter_25000_loss_0.01378_lr_0.000207',
},
{
    'snapshot': 'iter_20000_loss_0.01427_lr_0.000268',
},
{
    'snapshot': 'iter_15000_loss_0.01892_lr_0.000328',
},
{
    'snapshot': 'iter_10000_loss_0.01726_lr_0.000386',
},
{
    'snapshot': 'iter_5000_loss_0.02411_lr_0.000443',
},
    ]
    # argss = [
    #     {
    #         'snapshot': 'iter_5000_loss_0.01517_lr_0.000000',
    #     }
    # ]
    for args in argss:
        main(args)