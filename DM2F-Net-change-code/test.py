# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import deltaE_ciede2000, rgb2lab

import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)
to_tensor = ToTensor()
ckpt_path = './ckpt'
#exp_name = 'RESIDE_ITS'
#exp_name = 'O-Haze'
save_path = './ckpt'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_19000_loss_0.04261_lr_0.000014',
    #'snapshot': 'iter_20000_loss_0.04687_lr_0.000000'
}

to_test = {
    #'SOTS': TEST_SOTS_ROOT,
    'O-Haze': OHAZE_ROOT,

}

to_pil = transforms.ToPILImage()


def main(args):
    name = 'O-Haze'
    #log_path = os.path.join('/mnt/result_test', args['snapshot'] + '.txt')
    #open(log_path, 'w+').write(name+ " "+ args['snapshot'] + '\n\n')
    k = 0
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'val')
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            # if len(args['snapshot']) > 0:
            #     print('load snapshot \'%s\' for testing' % args['snapshot'])
            #     #net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
            #     net.load_state_dict(torch.load(os.path.join('/mnt/RESIDE_ITS/nochange/', args['snapshot'] + '.pth')))
            print('load snapshot for testing')
            net.load_state_dict(torch.load(os.path.join('/mnt/O-Haze/iter_20000_loss_0.04687_lr_0.000000.pth')))
            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims,ciede2000s = [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                # check_mkdir(os.path.join(save_path, exp_name,
                #                          '(%s)%s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
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
                    ciede2000 = deltaE_ciede2000(labg, labr).mean()
                    ciede2000s.append(ciede2000)
                    # print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}'
                    #       .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, ciede2000))
                    
                    # log = "%s: PSNR %.6f SSIM %.6f, CIEDE2000 %.6f" % (fs[i], psnr, ssim, ciede2000)
                    # open(log_path, 'a').write(log + '\n')
                # for r, f in zip(res.cpu(), fs):
                #     dehaze_path = os.path.join(ckpt_path, exp_name,
                #                      '(%s)%s_%s' % (exp_name, name, args['snapshot']), '%s.jpg' % f)
                #     to_pil(r).save(dehaze_path)
                    
                    # dehaze = Image.open(dehaze_path)
                    # dehaze = to_tensor(dehaze).numpy().transpose([1, 2, 0])
                    # # print(f'r{r.numpy().transpose([1, 2, 0])}')
                    # # print(f'dehaze{dehaze}')
                    # gt_path = '/mnt/data/O-Haze/val/gt/'+f+'.jpg'
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


if __name__ == '__main__':
#     argss = [{
#     'snapshot': 'iter_4000_loss_0.04839_lr_0.000164'
# }, {
#     'snapshot': 'iter_8000_loss_0.04875_lr_0.000126'
# }, {
#     'snapshot': 'iter_12000_loss_0.04679_lr_0.000088'
# },{
#     'snapshot': 'iter_16000_loss_0.04714_lr_0.000047'
# },{
#     'snapshot': 'iter_20000_loss_0.04687_lr_0.000000'
# }

#     ]
    # for args in argss:
    argss = [{
    'snapshot': 'iter_40000_loss_0.01183_lr_0.000000'
}, ]
    main(argss)
