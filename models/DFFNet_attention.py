from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from .submodule import *
from models.featExactor2 import FeatExactor
# ↑ 위에서 정의한 SelfAttentionAcrossFocus를 import 해야 한다면
# from .attention_module import SelfAttentionAcrossFocus 
# 처럼 별도 파일로 분리하는 것도 가능

class SelfAttentionAcrossFocus(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        # batch_first 인자 없이 MultiheadAttention 생성
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads
        )

    def forward(self, x):
        """
        x: (B, C, N, H, W)
        목표: N을 시퀀스 길이로 보고, 각 픽셀 위치에 대해 attention 적용.
        """
        B, C, N, H, W = x.shape
        # (B, C, N, H, W) -> (N, B, H, W, C)
        x = x.permute(2, 0, 3, 4, 1).contiguous()
        # reshape: (N, B*H*W, C)
        x = x.view(N, B * H * W, C)
        # Multi-head attention (입력 shape: (seq_len, batch, embed_dim))
        attended, _ = self.mha(x, x, x)
        # 복원: (N, B*H*W, C) -> (N, B, H, W, C)
        attended = attended.view(N, B, H, W, C)
        # (N, B, H, W, C) -> (B, C, N, H, W)
        attended = attended.permute(1, 4, 0, 2, 3).contiguous()
        return attended

class DFFNet(nn.Module):
    def __init__(self, clean, level=1, use_diff=1):
        """
        use_diff=1을 'Attention 사용 모드'로 활용 (기존 차분 대신).
        필요에 따라 별도 인자를 두어 분기 처리하는 것도 가능.
        """
        super(DFFNet, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()  # Encoder
        self.level = level
        self.use_diff = use_diff

        assert 1 <= level <= 4
        assert use_diff in [0, 1]

        # Attention 모듈 추가 (vol4는 channel=128, vol3=64, vol2=32, vol1=16 가정)
        # 실제 FeatExactor의 출력 채널 수에 맞춰야 함
        self.attn4 = SelfAttentionAcrossFocus(channels=128, num_heads=4)
        self.attn3 = SelfAttentionAcrossFocus(channels=64, num_heads=4)
        self.attn2 = SelfAttentionAcrossFocus(channels=32, num_heads=4)
        self.attn1 = SelfAttentionAcrossFocus(channels=16, num_heads=4)

        # DecoderBlocks
        if level == 1:
            self.decoder3 = decoderBlock(2, 16, 16, stride=(1,1,1), up=False, nstride=1)
        elif level == 2:
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1,1,1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 32, 32, up=True)
        elif level == 3:
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 64, 32, up=True)
            self.decoder5 = decoderBlock(2, 64, 64, up=True, pool=True)
        else:  # level == 4
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 64, 32, up=True)
            self.decoder5 = decoderBlock(2, 128, 64, up=True, pool=True)
            self.decoder6 = decoderBlock(2, 128, 128, up=True, pool=True)

        # disparity regression
        self.disp_reg = disparityregression(1)

    def forward(self, stack, focal_dist):
        """
        stack shape: (B, N, C, H, W)
         - B: batch
         - N: number of focus frames
         - C: channel (RGB=3 등)
         - H, W: height, width
        """
        B, N, C, H, W = stack.shape

        # 1) Encoder
        input_stack = stack.reshape(B*N, C, H, W)
        conv4, conv3, conv2, conv1 = self.feature_extraction(input_stack)
        # conv4 -> (B*N, 128, H/32, W/32)
        # conv3 -> (B*N, 64,  H/16, W/16)
        # conv2 -> (B*N, 32,  H/8,  W/8)
        # conv1 -> (B*N, 16,  H/4,  W/4)

        # 2) (B, N, C', H', W') 형태로 리쉐이프하여 3D volume으로
        vol4 = conv4.reshape(B, N, -1, H//32, W//32).permute(0, 2, 1, 3, 4)
        vol3 = conv3.reshape(B, N, -1, H//16, W//16).permute(0, 2, 1, 3, 4)
        vol2 = conv2.reshape(B, N, -1, H//8,  W//8 ).permute(0, 2, 1, 3, 4)
        vol1 = conv1.reshape(B, N, -1, H//4,  W//4 ).permute(0, 2, 1, 3, 4)
        # vol4 shape: (B, 128, N, H/32, W/32)
        # vol3 shape: (B, 64,  N, H/16, W/16)
        # ...

        # 3) Attention을 통한 cost volume 생성
        #    use_diff == 1이면 attention 적용, 아니면 그냥 사용
        if self.use_diff == 1:
            vol4 = self.attn4(vol4)  # (B, 128, N, H/32, W/32)
            vol3 = self.attn3(vol3)  # (B, 64,  N, ...)
            vol2 = self.attn2(vol2)
            vol1 = self.attn1(vol1)
        # else:
        #     그대로 vol4, vol3, vol2, vol1 사용

        # 4) Decoder
        if self.level == 1:
            _, cost3 = self.decoder3(vol1)

        elif self.level == 2:
            feat4_2x, cost4 = self.decoder4(vol2)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        elif self.level == 3:
            feat5_2x, cost5 = self.decoder5(vol3)
            feat4 = torch.cat((feat5_2x, vol2), dim=1)
            feat4_2x, cost4 = self.decoder4(feat4)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        else:  # level == 4
            feat6_2x, cost6 = self.decoder6(vol4)
            feat5 = torch.cat((feat6_2x, vol3), dim=1)
            feat5_2x, cost5 = self.decoder5(feat5)
            feat4 = torch.cat((feat5_2x, vol2), dim=1)
            feat4_2x, cost4 = self.decoder4(feat4)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        # 5) 최종 출력 해상도에 맞춰 upsample (bilinear)
        cost3 = F.interpolate(cost3, [H, W], mode='bilinear')
        pred3, std3 = self.disp_reg(F.softmax(cost3, 1), focal_dist, uncertainty=True)

        # 반환값들 설정
        stacked = [pred3]
        stds = [std3]

        if self.training:
            # level>=2 등에서 cost4, cost5, ... 추가 supervision
            if self.level >= 2:
                cost4 = F.interpolate(cost4, [H, W], mode='bilinear')
                pred4, std4 = self.disp_reg(F.softmax(cost4, 1), focal_dist, uncertainty=True)
                stacked.append(pred4)
                stds.append(std4)
                if self.level >= 3:
                    cost5 = F.interpolate(cost5.unsqueeze(1), [focal_dist.shape[1], H, W], 
                                         mode='trilinear').squeeze(1)
                    pred5, std5 = self.disp_reg(F.softmax(cost5, 1), focal_dist, uncertainty=True)
                    stacked.append(pred5)
                    stds.append(std5)
                    if self.level >= 4:
                        cost6 = F.interpolate(cost6.unsqueeze(1), [focal_dist.shape[1], H, W], 
                                             mode='trilinear').squeeze(1)
                        pred6, std6 = self.disp_reg(F.softmax(cost6, 1), focal_dist, uncertainty=True)
                        stacked.append(pred6)
                        stds.append(std6)

            return stacked, stds, None
        else:
            return pred3, torch.squeeze(std3), F.softmax(cost3, 1).squeeze()
