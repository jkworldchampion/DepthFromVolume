from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import time
from models import DFFNet
from utils import logger, write_log
torch.backends.cudnn.benchmark=True
from glob import glob
import logging
import numpy as np
import skimage.filters as skf
from torch.utils.data import DataLoader

# 표준시 설정
os.environ['TZ'] = 'Asia/Seoul'
time.tzset()

'''
Main code for Ours-FV and Ours-DFV training 
'''


parser = argparse.ArgumentParser(description='DFVDFF')
# === dataset =====
parser.add_argument('--dataset', default=['FoD500','DDFF12'], nargs='+',  help='data Name')
parser.add_argument('--DDFF12_pth', default=None, help='DDFF12 data path')
parser.add_argument('--FoD_pth', default=None, help='FOD data path')
parser.add_argument('--FoD_scale', default=0.2,
                    help='FoD dataset gt scale for loss balance, because FoD_GT: 0.1-1.5, DDFF12_GT 0.02-0.28, '
                         'empirically we find this scale help improve the model performance for our method and DDFF')
# ==== hypo-param =========
parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10]')
parser.add_argument('--level', type=int ,default=4, help='num of layers in network, please take a number in [1, 4]')
parser.add_argument('--use_diff', default=1, type=int, choices=[0,1], help='if use differential feat, 0: None,  1: diff cost volume')
parser.add_argument('--lvl_w', nargs='+', default=[8./15, 4./15, 2./15, 1./15],  help='for std weight')

parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--batchsize', type=int, default=20, help='samples per batch')

# ====== log path ==========
parser.add_argument('--loadmodel', default=None,   help='path to pre-trained checkpoint if any')
parser.add_argument('--savemodel', default=None, help='save path')
parser.add_argument('--seed', type=int, default=2021, metavar='S',  help='random seed (default: 2021)')

args = parser.parse_args()
args.logname = '_'.join(args.dataset)


# ============ init ===============
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

start_epoch = 1
best_loss = 1e5
total_iter = 0

model = DFFNet(clean=False,level=args.level, use_diff=args.use_diff)
model = nn.DataParallel(model)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# ========= load model if any ================
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()  } #if ('disp' not in k)
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    if 'epoch' in pretrained_dict:
        start_epoch = pretrained_dict['epoch']

    if 'iters' in pretrained_dict:
        total_iter = pretrained_dict['iters']

    if 'best' in pretrained_dict:
        best_loss = pretrained_dict['best']

#     if 'optimize' in pretrained_dict:
#         optimizer.load_state_dict(pretrained_dict['optimize'])

    print('load model from {}, start epoch {}, best_loss {}'.format(args.loadmodel, start_epoch, best_loss))

# 위치 변경
# msg = 'Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
# print(msg)
# logging.info(msg)


# ============ data loader ==============
#Create data loader

from dataloader import DDFF12Loader
database = '/data/DFF/my_ddff_trainVal.h5' if args.DDFF12_pth is None else  args.DDFF12_pth
# ─── train+val 전체 데이터를 학습에 사용 ───
DDFF12_train = DDFF12Loader(
    database,
    stack_key="stack_train",
    disp_key="disp_train",
    n_stack=args.stack_num,
    min_disp=0.02,
    max_disp=0.28,
    b_test=False
)
DDFF12_val = DDFF12Loader(
    database,
    stack_key="stack_val",
    disp_key="disp_val",
    n_stack=args.stack_num,
    min_disp=0.02,
    max_disp=0.28,
    b_test=True
)

# train에는 train+val 모두, val에는 val만
base_train_ds = torch.utils.data.ConcatDataset([DDFF12_train, DDFF12_val])
# base_train_ds = torch.utils.data.ConcatDataset([DDFF12_train])
base_val_ds   = torch.utils.data.ConcatDataset([DDFF12_val])

class ResizeStackDataset(Dataset):
    def __init__(self, base_ds, size=(224,224)):
        self.base_ds = base_ds
        self.size = size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        # loader가 (img_stack, gt_disp, foc_dist) 세 튜플을 반환한다고 가정
        img_stack, gt_disp, foc_dist = self.base_ds[idx]

        # img_stack: Tensor of shape [n_stack, C, H, W]
        # → 바로 interpolate로 (224,224)로 리사이즈
        img_stack = F.interpolate(
            img_stack,
            size=self.size,
            mode='bilinear',
            align_corners=False
        )

        return img_stack, gt_disp, foc_dist

# 래핑된 dataset
dataset_train = ResizeStackDataset(base_train_ds, size=(224,224))
dataset_val   = ResizeStackDataset(base_val_ds,   size=(224,224))

# ============ DataLoader 생성 ============
TrainImgLoader = DataLoader(
    dataset=dataset_train,
    num_workers=4,
    batch_size=args.batchsize,
    shuffle=True,
    drop_last=True
)
ValImgLoader = DataLoader(
    dataset=dataset_val,
    num_workers=1,
    batch_size=12,
    shuffle=False,
    drop_last=True
)


print('%d batches per epoch'%(len(TrainImgLoader)))
# =========== Train func. =========
def train(img_stack_in, disp, foc_dist):
    model.train()
    img_stack_in   = Variable(torch.FloatTensor(img_stack_in))
    gt_disp    = Variable(torch.FloatTensor(disp))
    img_stack, gt_disp, foc_dist = img_stack_in.cuda(),  gt_disp.cuda(), foc_dist.cuda()

    #---------
    max_val = torch.where(foc_dist>=100, torch.zeros_like(foc_dist), foc_dist) # exclude padding value
    min_val = torch.where(foc_dist<=0, torch.ones_like(foc_dist)*10, foc_dist)  # exclude padding value
    mask = (gt_disp >= min_val.min(dim=1)[0].view(-1,1,1,1)) & (gt_disp <= max_val.max(dim=1)[0].view(-1,1,1,1)) #
    mask.detach_()
    #----

    optimizer.zero_grad()
    beta_scale = 1 # smooth l1 do not have beta in 1.6, so we increase the input to and then scale back -- no significant improve according to our trials
    stacked, stds, _ = model(img_stack, foc_dist)


    loss = 0
    for i, (pred, std) in enumerate(zip(stacked, stds)):
        _cur_loss = F.smooth_l1_loss(pred[mask] * beta_scale, gt_disp[mask]* beta_scale, reduction='none') / beta_scale
        loss = loss + args.lvl_w[i] * _cur_loss.mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    vis = {}
    vis['pred'] = stacked[0].detach().cpu()
    vis['mask'] = mask.type(torch.float).detach().cpu()
    lossvalue = loss.data

    del stacked
    del loss
    return lossvalue,vis

# ───────── metrics 계산용 함수 ─────────
def calmetrics(pred, target, mse_factor=1.0, accthrs=(1.25, 1.25**2, 1.25**3),
               bumpinessclip=0.05, ignore_zero=True):
    """Return a (1×10) numpy array:
       [MSE, RMS, log RMS, Abs_rel, Sqr_rel, a1, a2, a3, bump, avgUnc]"""
    metrics = np.zeros((1, 10), dtype=float)
    if target.sum() == 0:
        return metrics

    pred_ = pred.copy()
    if ignore_zero:
        pred_[target == 0.0] = 0.0
        numPixels = (target > 0.0).sum()
    else:
        numPixels = target.size

    # ① MSE & ② RMS
    mse = np.square(pred_ - target).sum() / numPixels * mse_factor
    metrics[0, 0] = mse
    metrics[0, 1] = np.sqrt(mse)

    # ③ log RMS
    logrms = (np.ma.log(pred_) - np.ma.log(target))
    metrics[0, 2] = np.sqrt((logrms**2).sum() / numPixels)

    # ④ Abs rel ⑤ Sqr rel
    mask = target > 0                    # 0 깊이(패딩) 제거
    valid_pred = pred_[mask]
    valid_gt   = target[mask]

    abs_rel = (np.abs(valid_pred - valid_gt) / valid_gt).mean()
    sqr_rel = (np.square(valid_pred - valid_gt) / valid_gt).mean()
    
    metrics[0, 3] = abs_rel
    metrics[0, 4] = sqr_rel

    # ⑥–⑧ 정확도 a1 a2 a3
    acc = np.maximum(pred_ / target, target / pred_)
    for i, thr in enumerate(accthrs):
        metrics[0, 5 + i] = (acc < thr).sum() / numPixels * 100.

    # ⑨ bumpiness (Frobenius-norm Hessian)
    diff = (pred - target).astype('float64')
    bump = 0.0
    for d in (skf.scharr_v(diff), skf.scharr_h(diff)):
        bump += np.sqrt(skf.scharr_v(d)**2 + skf.scharr_h(d)**2)
    bump = np.clip(bump, 0, bumpinessclip)
    metrics[0, 8] = bump[target > 0].sum() / numPixels * 100.

    # ⑩ avgUnc → 나중에 std.mean()으로 덮어씀
    return metrics


def valid(img_stack_in,disp, foc_dist):
    model.eval()
    img_stack = Variable(torch.FloatTensor(img_stack_in))
    gt_disp = Variable(torch.FloatTensor(disp))
    img_stack, gt_disp, foc_dist = img_stack.cuda() , gt_disp.cuda(), foc_dist.cuda()

    #---------
    mask = gt_disp > 0
    mask.detach_()
    #----
    with torch.no_grad():
        pred_disp, _, _ = model(img_stack, foc_dist)
        loss = (F.mse_loss(pred_disp[mask] , gt_disp[mask] , reduction='mean')) # use MSE loss for val

    vis = {}
    vis['mask'] = mask.type(torch.float).detach().cpu()
    vis["pred"] = pred_disp.detach().cpu()

    return loss, vis



def adjust_learning_rate(optimizer, epoch):
    # turn out we do not need adjust lr, the results is already good enough
    if epoch <= args.epochs:
        lr = args.lr
    else:
        lr = args.lr * 0.1 #1e-5  will not used in this project
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def main():
    global start_epoch, best_loss, total_iter
    saveName = args.logname + "_ep{}_b{}_full".format(
        args.epochs, args.batchsize)
    if args.use_diff > 0:
        saveName = saveName + '_diff{}_from_DFF-DFV'.format(args.use_diff)

    # log 및 model 저장 폴더 생성
    save_folder = os.path.join(os.path.abspath(args.savemodel), saveName)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # logging 설정: log 파일은 save_folder 아래 train_log.txt로 저장
    log_file_path = os.path.join(save_folder, "train_log.txt")
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 이게 여기 있어야 되겠지?
    msg = 'Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print(msg)
    logging.info(msg)

    # logger 객체 생성
    train_log = logger.Logger(os.path.abspath(args.savemodel), name=saveName + '/train')
    val_log = logger.Logger(os.path.abspath(args.savemodel), name=saveName + '/val')

    total_iters = total_iter

    # 전체 학습 시작 시각 기록 (ETA 계산용)
    global_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        total_train_loss = 0
        lr_ = adjust_learning_rate(optimizer, epoch)
        train_log.scalar_summary('lr_epoch', lr_, epoch)

        # 에폭 시작 메시지
        msg = "=== Epoch {} / {} ===".format(epoch, args.epochs)
        print(msg)
        logging.info(msg)

        ## Training ##
        for batch_idx, (img_stack, gt_disp, foc_dist) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, vis = train(img_stack, gt_disp, foc_dist)

            if total_iters % 10 == 0:
                torch.cuda.synchronize()
                msg = 'epoch {}:  {}/{} train_loss = {:.6f} , time = {:.2f}'.format(
                    epoch, batch_idx, len(TrainImgLoader), loss, time.time() - start_time)
                print(msg)
                logging.info(msg)
                train_log.scalar_summary('loss_batch', loss, total_iters)  # 원래 total_iters에서 epoch로 수정해야함.

            total_train_loss += loss
            total_iters += 1

        # 기록용 마지막 배치 로그 (write_log 함수 호출)
        write_log(vis, img_stack[:, 0], img_stack[:, -1], gt_disp, train_log, epoch, thres=0.05)
        avg_loss = total_train_loss / len(TrainImgLoader)
        train_log.scalar_summary('avg_loss', avg_loss, epoch)
        msg = "Epoch {} Average Train Loss: {:.6f}".format(epoch, avg_loss)
        print(msg)
        logging.info(msg)

        # 모델 저장
        checkpoint_path = os.path.join(save_folder, 'model_{}.tar'.format(epoch))
        torch.save({
            'epoch': epoch + 1,
            'iters': total_iters + 1,
            'best': best_loss,
            'state_dict': model.state_dict(),
            'optimize': optimizer.state_dict(),
        }, checkpoint_path)

        # 최근 5개의 체크포인트만 유지
        list_ckpt = glob(os.path.join(save_folder, 'model_*'))
        list_ckpt.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if len(list_ckpt) > 5:
            os.remove(list_ckpt[0])

        if epoch % 5 == 0:
            total_val_loss = 0
            accthrs = (1.25, 1.25**2, 1.25**3)
            avgmetrics = np.zeros((1, 10), dtype=float)
            sample_cnt = 0

            for batch_idx, (img_stack, gt_disp, foc_dist) in enumerate(ValImgLoader):
                with torch.no_grad():
                    start_time = time.time()
                    # 평균 손실계산
                    val_loss, viz = valid(img_stack, gt_disp, foc_dist)

                    # 추가: metric 계산 ------------------------------
                    pred_disp, std, _ = model(img_stack.cuda(), foc_dist.cuda())
                    for b in range(pred_disp.size(0)):
                        pd = pred_disp[b].squeeze().detach().cpu().numpy()
                        gt = gt_disp[b].squeeze().cpu().numpy()
                        m = calmetrics(pd, gt, 1.0, accthrs)
                        m[0, -1] = std[b].mean().detach().cpu().numpy()  # avgUnc
                        avgmetrics += m
                        sample_cnt += 1
                    # -----------------------------------------------

                if batch_idx % 10 == 0:
                    torch.cuda.synchronize()
                    msg = ('[val] epoch {} : {}/{} val_loss = {:.6f} , time = {:.2f}'
                           .format(epoch, batch_idx, len(ValImgLoader),
                                   val_loss, time.time() - start_time))
                    print(msg); logging.info(msg)
                total_val_loss += val_loss

            avg_val_loss = total_val_loss / len(ValImgLoader)
            avgmetrics   = avgmetrics / sample_cnt          # 10 종 평균
            write_log(viz, img_stack[:,0], img_stack[:,-1], gt_disp,
                      val_log, epoch, thres=0.05)
            val_log.scalar_summary('avg_loss', avg_val_loss, epoch)
            # 항상 metric 찍기
            names = ["MSE","RMS","logRMS","Abs_rel","Sqr_rel",
                          "a1","a2","a3","bump","avgUnc"]
            metric_str = " | ".join(f"{n:>7}" for n in names)
            values_str = " | ".join(f"{v:7.5f}" for v in avgmetrics.flatten())
            msg_metrics = f"[VAL @ epoch {epoch}] {metric_str}\n{values_str}"
            print(msg_metrics)
            logging.info(msg_metrics)

            # === best 모델 & metric 저장 ===
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch':   epoch + 1,
                    'iters':   total_iters + 1,
                    'best':    best_loss,
                    'state_dict': model.state_dict(),
                    'optimize':   optimizer.state_dict(),
                }, os.path.join(save_folder, 'best.tar'))

        torch.cuda.empty_cache()

        # 현재 에폭 소요 시간 및 ETA 계산
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - global_start_time
        remaining_epochs = args.epochs - epoch
        avg_epoch_time = elapsed_time / epoch  # epoch당 평균 소요 시간
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        eta_msg = "Estimated training time remaining: {}".format(eta_str)
        print(eta_msg)
        logging.info(eta_msg)

if __name__ == '__main__':
    main()
