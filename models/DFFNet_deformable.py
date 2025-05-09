from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from .submodule import *
from models.featExactor2 import FeatExactor
# DeformConv2d가 submodule에 구현되어 있다고 가정합니다.
from .deform_conv import DeformConv2d

class DeformableFusionAcrossFocus(nn.Module):
    """
    입력 텐서 x: (B, C, N, H, W)
    - B: 배치 크기
    - C: 채널 수
    - N: 포커스 스택 프레임 수
    - H, W: 공간 해상도
    이 때, N 차원을 2D convolution의 너비 축으로 취급하여 deformable conv를 적용한 후,
    원래 shape (B, C, N, H, W)로 복원합니다.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # offset을 예측하는 convolution: 출력 채널 수는 2 * kernel_size (x, y offset)
        self.offset_conv = nn.Conv2d(channels, 2 * kernel_size, kernel_size=(1, kernel_size), 
                                     padding=(0, kernel_size//2))
        # Deformable convolution layer
        self.deform_conv = DeformConv2d(channels, channels, kernel_size=(1, kernel_size), 
                                        padding=(0, kernel_size//2))

    def forward(self, x):
        # x: (B, C, N, H, W)
        B, C, N, H, W = x.shape
        # Permute: (B, H, W, C, N)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        # Reshape: merge (B, H, W) into batch dimension -> (B*H*W, C, 1, N)
        x = x.view(B * H * W, C, 1, N)
        # 예측 offset: (B*H*W, 2*kernel_size, 1, N)
        offset = self.offset_conv(x)
        # deformable convolution 적용: 결과 (B*H*W, C, 1, N)
        x = self.deform_conv(x, offset)
        # 원래 shape 복원: (B, H, W, C, N)
        x = x.view(B, H, W, C, N)
        # Permute back: (B, C, N, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x

class DFFNet(nn.Module):
    def __init__(self, clean, level=1, use_diff=1):
        """
        use_diff=1이면 deformable convolution을 사용하여 포커스 스택 간 fusion을 수행합니다.
        """
        super(DFFNet, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()  # Encoder
        self.level = level
        self.use_diff = use_diff

        assert 1 <= level <= 4
        assert use_diff in [0, 1]

        # Deformable fusion 모듈 추가 (출력 채널 수는 FeatExactor의 출력에 맞춰 조정)
        self.deform_fuse4 = DeformableFusionAcrossFocus(channels=128, kernel_size=3)
        self.deform_fuse3 = DeformableFusionAcrossFocus(channels=64,  kernel_size=3)
        self.deform_fuse2 = DeformableFusionAcrossFocus(channels=32,  kernel_size=3)
        self.deform_fuse1 = DeformableFusionAcrossFocus(channels=16,  kernel_size=3)

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
        else:
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 64, 32, up=True)
            self.decoder5 = decoderBlock(2, 128, 64, up=True, pool=True)
            self.decoder6 = decoderBlock(2, 128, 128, up=True, pool=True)

        # disparity regression
        self.disp_reg = disparityregression(1)

    def forward(self, stack, focal_dist):
        """
        stack: (B, N, C, H, W)
         - B: 배치 크기, N: 포커스 스택 프레임 수, C: 채널, H, W: 공간 해상도
        """
        B, N, C, H, W = stack.shape

        # 1) Encoder
        input_stack = stack.reshape(B * N, C, H, W)
        conv4, conv3, conv2, conv1 = self.feature_extraction(input_stack)
        # conv4: (B*N, 128, H/32, W/32)
        # conv3: (B*N, 64,  H/16, W/16)
        # conv2: (B*N, 32,  H/8,  W/8)
        # conv1: (B*N, 16,  H/4,  W/4)

        # 2) 3D volume로 재구성: (B, N, C', H', W') -> permute -> (B, C', N, H', W')
        vol4 = conv4.reshape(B, N, -1, H//32, W//32).permute(0, 2, 1, 3, 4)
        vol3 = conv3.reshape(B, N, -1, H//16, W//16).permute(0, 2, 1, 3, 4)
        vol2 = conv2.reshape(B, N, -1, H//8,  W//8).permute(0, 2, 1, 3, 4)
        vol1 = conv1.reshape(B, N, -1, H//4,  W//4).permute(0, 2, 1, 3, 4)
        # 각 volume shape: (B, C', N, H', W')

        # 3) Deformable Fusion 적용 (use_diff==1이면 deformable fusion 사용)
#         if self.use_diff == 1:
        # 그냥 deformable 무조건 적용
        vol4 = self.deform_fuse4(vol4)  # (B, 128, N, H/32, W/32)
        vol3 = self.deform_fuse3(vol3)  # (B, 64,  N, H/16, W/16)
        vol2 = self.deform_fuse2(vol2)  # (B, 32,  N, H/8,  W/8)
        vol1 = self.deform_fuse1(vol1)  # (B, 16,  N, H/4,  W/4)

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
        else:
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

        # Training 시 다중 스케일 예측값 반환
        stacked = [pred3]
        stds = [std3]
        if self.training:
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
