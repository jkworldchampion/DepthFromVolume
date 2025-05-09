import argparse
import os
import time
import numpy as np
import skimage.filters as skf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import DFFNet
from dataloader import DDFF12Loader


def calmetrics(pred, target, mse_factor=1.0, accthrs=(1.25, 1.25**2, 1.25**3), bumpinessclip=0.05):
    """
    Compute evaluation metrics:
    [MSE, RMS, logRMS, Abs_rel, Sqr_rel, a1, a2, a3, bump, avgUnc]
    """
    metrics = np.zeros((1, 10), dtype=float)
    # filter out padded zeros
    mask = target > 0
    if mask.sum() == 0:
        return metrics
    numPixels = mask.sum()

    # ① MSE & ② RMS
    diff = pred - target
    mse = np.square(diff[mask]).sum() / numPixels * mse_factor
    metrics[0, 0] = mse
    metrics[0, 1] = np.sqrt(mse)

    # ③ log RMS
    logrms = np.log(pred[mask]) - np.log(target[mask])
    metrics[0, 2] = np.sqrt((logrms**2).sum() / numPixels)

    # ④ Abs_rel & ⑤ Sqr_rel
    valid_pred = pred[mask]
    valid_gt = target[mask]
    metrics[0, 3] = np.mean(np.abs(valid_pred - valid_gt) / valid_gt)
    metrics[0, 4] = np.mean(np.square(valid_pred - valid_gt) / valid_gt)

    # ⑥–⑧ accuracies a1, a2, a3
    acc = np.maximum(pred / target, target / pred)
    for i, thr in enumerate(accthrs):
        metrics[0, 5 + i] = np.sum(acc[mask] < thr) / numPixels * 100.0

    # ⑨ bumpiness (Frobenius norm of Hessian)
    bump = np.zeros_like(diff)
    for d in (skf.scharr_v(diff), skf.scharr_h(diff)):
        bump += np.sqrt(skf.scharr_v(d)**2 + skf.scharr_h(d)**2)
    bump = np.clip(bump, 0, bumpinessclip)
    metrics[0, 8] = np.sum(bump[mask]) / numPixels * 100.0

    # ⑩ avgUnc: placeholder (to be overwritten by std.mean())
    return metrics


def main(image_size=(383, 552)):
    parser = argparse.ArgumentParser(description='DFV Evaluation')
    parser.add_argument('--data_path', default='./data/DFF/my_ddff_trainVal.h5', help='DDFF12 data file')
    parser.add_argument('--loadmodel', required=True, help='path to trained checkpoint')
    parser.add_argument('--outdir', default='./DDFF12_eval/', help='output directory')
    parser.add_argument('--max_disp', type=float, default=0.28)
    parser.add_argument('--min_disp', type=float, default=0.02)
    parser.add_argument('--stack_num', type=int, default=5)
    parser.add_argument('--use_diff', type=int, default=1)
    parser.add_argument('--level', type=int, default=4)
    args = parser.parse_args()

    # Load model
    model = DFFNet(clean=False, level=args.level, use_diff=args.use_diff)
    model = nn.DataParallel(model).cuda()
    ckpt = torch.load(args.loadmodel)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    # Prepare transforms & dataloader
    pad_h = int(np.ceil(image_size[0] / 32) * 32)
    pad_w = int(np.ceil(image_size[1] / 32) * 32)
    transform_test = transforms.Compose([
        DDFF12Loader.ToTensor(),
        DDFF12Loader.PadSamples((pad_h, pad_w)),
    ])
    test_set = DDFF12Loader(
        args.data_path,
        stack_key='stack_val',
        disp_key='disp_val',
        transform=transform_test,
        n_stack=args.stack_num,
        min_disp=args.min_disp,
        max_disp=args.max_disp,
        b_test=True
    )
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # Metrics accumulation
    accthrs = (1.25, 1.25**2, 1.25**3)
    avgmetrics = np.zeros((1, 10), dtype=float)
    sample_cnt = 0
    time_rec = []

    # Evaluation loop
    with torch.no_grad():
        for idx, (img_stack, disp, foc_dist) in enumerate(dataloader):
            if idx % 10 == 0:
                print(f'Processing sample {idx}/{len(dataloader)}')

            img_stack = img_stack.cuda()
            gt_disp = disp.cuda()

            torch.cuda.synchronize()
            start_t = time.time()
            raw_pred, std, _ = model(img_stack, foc_dist.cuda())
            torch.cuda.synchronize()
            time_rec.append(time.time() - start_t)

            # Convert to numpy (remove batch and channel dims)
            pred_np = raw_pred.squeeze().cpu().numpy()
            gt_np = gt_disp.squeeze().cpu().numpy()

            # Compute metrics
            m = calmetrics(pred_np, gt_np, mse_factor=1.0, accthrs=accthrs)
            m[0, -1] = std.mean().cpu().item()  # avgUnc
            avgmetrics += m
            sample_cnt += 1

    # Final aggregation and print
    avgmetrics /= sample_cnt
    names = ["MSE", "RMS", "logRMS", "Abs_rel", "Sqr_rel", "a1", "a2", "a3", "bump", "avgUnc"]
    print('\n========= Final Evaluation Results =========')
    print(' | '.join(f"{n:>7}" for n in names))
    print(' | '.join(f"{v:7.5f}" for v in avgmetrics.flatten()))
    print(f'Mean inference time (s): {np.mean(time_rec):.4f}')


if __name__ == '__main__':
    main()
