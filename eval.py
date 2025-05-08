import argparse
import cv2
from models import DFFNet
import numpy as np
import os
import skimage.filters as skf
import time
from models.submodule import *

import  matplotlib
# matplotlib.use('TkAgg') # 그래픽이 없는 환경이라
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision


'''
Code for Ours-FV and Ours-DFV evaluation on DDFF-12 dataset  
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--data_path', default='./data/DFF/my_ddff_trainVal.h5',help='test data path')
parser.add_argument('--loadmodel', default=None, help='model path')
parser.add_argument('--outdir', default='./DDFF12/',help='output dir')

parser.add_argument('--max_disp', type=float ,default=0.28, help='maxium disparity')
parser.add_argument('--min_disp', type=float ,default=0.02, help='minium disparity')

parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
parser.add_argument('--use_diff', default=1, choices=[0,1], help='if use differential images as input, change it according to the loaded checkpoint!')

parser.add_argument('--level', type=int, default=4, help='num of layers in network, please take a number in [1, 4]')
args = parser.parse_args()

# !!! Only for users who download our pre-trained checkpoint, comment the next four line if you are not !!!
# if os.path.basename(args.loadmodel) == 'DFF-DFV.tar' :
#     args.use_diff = 1
# else:
#     args.use_diff = 0

# dataloader
from dataloader import DDFF12Loader

# construct model
model = DFFNet(clean=False,level=args.level, use_diff=args.use_diff)
model = nn.DataParallel(model)
model.cuda()
ckpt_name = os.path.basename(os.path.dirname(args.loadmodel))

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
#     pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def calmetrics( pred, target, mse_factor, accthrs, bumpinessclip=0.05, ignore_zero=True):
    metrics = np.zeros((1, 7 + len(accthrs)), dtype=float)

    if target.sum() == 0:
        return metrics

    pred_ = np.copy(pred)
    if ignore_zero:
        pred_[target == 0.0] = 0.0
        numPixels = (target > 0.0).sum()  # number of valid pixels
    else:
        numPixels = target.size

    # euclidean norm
    metrics[0, 0] = np.square(pred_ - target).sum() / numPixels * mse_factor

    # RMS
    metrics[0, 1] = np.sqrt(metrics[0, 0])

    # log RMS
    logrms = (np.ma.log(pred_) - np.ma.log(target))
    metrics[0, 2] = np.sqrt(np.square(logrms).sum() / numPixels)

    # absolute relative
    metrics[0, 3] = np.ma.divide(np.abs(pred_ - target), target).sum() / numPixels

    # square relative
    metrics[0, 4] = np.ma.divide(np.square(pred_ - target), target).sum() / numPixels

    # accuracies
    acc = np.ma.maximum(np.ma.divide(pred_, target), np.ma.divide(target, pred_))
    for i, thr in enumerate(accthrs):
        metrics[0, 5 + i] = (acc < thr).sum() / numPixels * 100.

    # badpix
    metrics[0, 8] = (np.abs(pred_ - target) > 0.07).sum() / numPixels * 100.

    # bumpiness -- Frobenius norm of the Hessian matrix
    diff = np.asarray(pred - target, dtype='float64')  # PRED or PRED_
    chn = diff.shape[2] if len(diff.shape) > 2 else 1
    bumpiness = np.zeros_like(pred_).astype('float')
    for c in range(0, chn):
        if chn > 1:
            diff_ = diff[:, :, c]
        else:
            diff_ = diff
        dx = skf.scharr_v(diff_)
        dy = skf.scharr_h(diff_)
        dxx = skf.scharr_v(dx)
        dxy = skf.scharr_h(dx)
        dyy = skf.scharr_h(dy)
        dyx = skf.scharr_v(dy)
        hessiannorm = np.sqrt(np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
        bumpiness += np.clip(hessiannorm, 0, bumpinessclip)
    bumpiness = bumpiness[target > 0].sum() if ignore_zero else bumpiness.sum()
    metrics[0, 9] = bumpiness / chn / numPixels * 100.

    return metrics


def main(image_size=(383, 552)):
    model.eval()

    # 1) 패딩 크기 계산 (32의 배수)
    test_pad_size = (
        int(np.ceil(image_size[0] / 32) * 32),
        int(np.ceil(image_size[1] / 32) * 32)
    )

    # 2) 전처리: ToTensor + PadSamples (학습 시 Normalize가 없었다면 여기서도 제거)
    transform_test = torchvision.transforms.Compose([
        DDFF12Loader.ToTensor(),
        DDFF12Loader.PadSamples(test_pad_size),
        # DDFF12Loader.Normalize(...)  # 학습 때 Normalize 안 썼다면 주석 유지
    ])

    # 3) 데이터로더 생성
    test_set = DDFF12Loader(
        args.data_path,
        stack_key="stack_val",
        disp_key="disp_val",
        transform=transform_test,
        n_stack=args.stack_num,
        min_disp=args.min_disp,
        max_disp=args.max_disp,
        b_test=True
    )
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # 4) 지표 준비
    accthrs = [1.25, 1.25**2, 1.25**3]
    avgmetrics = np.zeros((1, 7 + len(accthrs) + 1), dtype=float)  # +1 for avgUnc
    test_num = len(dataloader)
    time_rec = np.zeros(test_num)

    # 5) 평가 루프
    for inx, (img_stack, disp, foc_dist) in enumerate(dataloader):
        if inx % 10 == 0:
            print(f'processing: {inx}/{test_num}')

        # GPU 로 이동
        img_stack = img_stack.cuda()
        gt_disp    = disp.cuda()

        # 5-1) 추론 및 시간 측정, 그리고 바로 NumPy 변환
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            raw_pred, std, focusMap = model(img_stack, foc_dist.cuda())
            torch.cuda.synchronize()
            time_rec[inx] = time.time() - start_time

            pred_disp = raw_pred.squeeze().cpu().numpy()   # (H_pad, W_pad)
            gt_disp_np = gt_disp.squeeze().cpu().numpy()   # (H_pad, W_pad)

        # ❶ 지표 계산 (리턴 shape: (1,10))
        metrics = calmetrics(
            pred_disp,
            gt_disp_np,
            mse_factor=1.0,
            accthrs=accthrs,
            bumpinessclip=0.05,
            ignore_zero=True
        )
        metrics = metrics.squeeze(0)  # now shape (10,)
        # 누적
        avgmetrics[0, :-1] += metrics
        avgmetrics[0, -1]  += std.mean().cpu().item()

        torch.cuda.empty_cache()

    # 6) 최종 결과 출력
    final_res = (avgmetrics / test_num)[0]
    final_res = np.delete(final_res, 8)  # badpix 컬럼 제거
    names = ["MSE", "RMS", "log RMS", "Abs_rel", "Sqr_rel",
             "a1", "a2", "a3", "bump", "avgUnc"]
    print('==============  Final result =================')
    print("  " + " | ".join(f"{n:>7}" for n in names))
    print("  " + "  ".join(f"{v:7.6f}" for v in final_res.tolist()))
    print('runtime mean:', np.mean(time_rec[1:]))  # 첫 워밍업 프레임 제외


if __name__ == '__main__':
    main()

