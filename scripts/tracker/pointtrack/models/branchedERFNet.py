import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import models.erfnet as erfnet

class LocationEmbedding(nn.Module):
    
    def __init__(
        self,
        device: torch.device,
        dim_g: int=64,
        wave_len: int=1000,
    ) -> None:
        
        super().__init__()
        
        feat_range = torch.arange(
            start=0,
            end=dim_g / 8,
            step=1,
            device=device,
        )
        
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))
        self.dim_mat = dim_mat.view(1, 1, -1)
        
    def forward(
        self,
        f_g: torch.Tensor,
    ) -> torch.Tensor:
        
        x_min, y_min, x_max, y_max = torch.chunk(
            f_g, 4,
            dim=1,
        )
        
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.
        
        position_mat = torch.cat(
            (cx, cy, w, h),
            dim=-1,
        )
        
        position_mat = position_mat.view(f_g.shape[0], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * self.dim_mat
        mul_mat = mul_mat.view(f_g.shape[0], -1)
        
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        
        embedding = torch.cat(
            (sin_mat, cos_mat),
            dim=-1,
        )
        
        return embedding

class BranchedERFNet(nn.Module):

    def __init__(
        self,
        num_classes: list,
        input_channel: int=3,
        encoder=None,
    ) -> None:

        super().__init__()

        if encoder is None:
            self.encoder = erfnet.Encoder(
                num_classes=sum(num_classes),
                input_channel=input_channel,
            )
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()

        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    @torch.no_grad()
    def init_output(
        self,
        n_sigma: int=1,
    ) -> None:

        output_conv = self.decoders[0].output_conv

        output_conv.weight[:, 0: 2, ...].fill_(0)
        output_conv.bias[0: 2].fill_(0)

        output_conv.weight[:, 2: 2 + n_sigma, ...].fill_(0)
        output_conv.bias[2: 2 + n_sigma].fill_(1)

    def forward(
        self,
        input: torch.Tensor,
        only_encode: bool=False,
    ) -> torch.Tensor:

        if only_encode:
            return self.encoder.forward(
                input=input,
                predict=True,
            )
        else:
            output = self.encoder(input)

        return torch.cat(
            tensors=[
                decoder.forward(output)
                for decoder in self.decoders
            ],
            dim=1,
        )


class PointFeatFuse3P(nn.Module):

    def __init__(
        self,
        num_points: int=250,
        ic: int=7,
        oc: int=64,
        maxpool: bool=True,
    ) -> None:

        super().__init__()

        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)

        self.e_conv1 = nn.Conv1d(ic, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)
        self.e_conv3 = nn.Conv1d(128, 256, 1)

        self.t_conv1 = nn.Conv1d(3, 64, 1)
        self.t_conv2 = nn.Conv1d(64, 128, 1)
        self.t_conv3 = nn.Conv1d(128, 128, 1)

        self.conv4 = nn.Conv1d(512, 256, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        self.conv6 = nn.Conv1d(512, oc, 1)

        self.maxpool = maxpool
        self.num_points = num_points

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        x = F.leaky_relu(self.conv1(x))
        emb = F.leaky_relu(self.e_conv1(emb))
        t = F.leaky_relu(self.t_conv1(t))

        x = F.leaky_relu(self.conv2(x))
        emb = F.leaky_relu(self.e_conv2(emb))
        t = F.leaky_relu(self.t_conv2(t))

        x = F.leaky_relu(self.conv3(x))
        emb = F.leaky_relu(self.e_conv3(emb))
        t = F.leaky_relu(self.t_conv3(t))

        x1 = F.leaky_relu(self.conv4(torch.cat((x, emb, t), dim=1)))
        x1 = F.leaky_relu(self.conv5(x1))
        x1 = F.leaky_relu(self.conv6(x1))

        if self.maxpool:
            x1 = F.max_pool1d(x1, self.num_points).squeeze(-1)
        else:
            x1 = F.avg_pool1d(x1, self.num_points).squeeze(-1)

        return x1

class PoseNetFeatOffsetEmb(nn.Module):

    def __init__(
        self,
        num_points: int=250,
        ic: int=7,
        border_ic: int=6,
        output_dim: int=64,
    ) -> None:

        super().__init__()

        bc = 256
        self.border_points = num_points // 3
        self.num_points = num_points - self.border_points

        self.borderConv = PointFeatFuse3P(
            ic=border_ic,
            oc=bc,
            num_points=self.border_points,
        )

        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.e_conv1 = nn.Conv1d(ic, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)
        self.e_conv3 = nn.Conv1d(128, 256, 1)
        self.e_conv1_bn = nn.BatchNorm1d(64)
        self.e_conv2_bn = nn.BatchNorm1d(128)
        self.e_conv3_bn = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(512, 256, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        self.conv6 = nn.Conv1d(512, 64, 1)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5_bn = nn.BatchNorm1d(512)

        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 512, 1)
        self.conv9 = nn.Conv1d(512, 64, 1)
        self.conv7_bn = nn.BatchNorm1d(256)
        self.conv8_bn = nn.BatchNorm1d(512)

        self.conv_weight = nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(704 + bc, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        inp: torch.Tensor,
        borders: torch.Tensor,
        spatialEmbs: torch.Tensor,
    ) -> torch.Tensor:

        x, emb = inp[:, inp.shape[1] - 2:], inp[:, :inp.shape[1] - 2]

        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        emb = F.leaky_relu(self.e_conv1_bn(self.e_conv1(emb)))

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        emb = F.leaky_relu(self.e_conv2_bn(self.e_conv2(emb)))

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))          # B,256,N
        emb = F.leaky_relu(self.e_conv3_bn(self.e_conv3(emb)))  # B,256,N

        pointfeat_2 = torch.cat(
            [x, emb], 
            dim=1,
        )

        x1 = F.leaky_relu(self.conv4_bn(self.conv4(pointfeat_2)))
        x1 = F.leaky_relu(self.conv5_bn(self.conv5(x1)))
        x1 = F.leaky_relu(self.conv6(x1))                       # B,64,N
        ap_x1 = F.avg_pool1d(x1, self.num_points).squeeze(-1)   # B,64

        x2 = F.leaky_relu(self.conv7_bn(self.conv7(pointfeat_2)))
        x2 = F.leaky_relu(self.conv8_bn(self.conv8(x2)))
        x2 = F.leaky_relu(self.conv9(x2))                       # B,64,N
        mp_x2 = F.max_pool1d(x2, self.num_points).squeeze(-1)   # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))   #B,1,N
        weight = F.softmax(weightFeat, dim=2)
        weight_x3 = (weight.expand_as(pointfeat_2) * pointfeat_2).sum(2)

        border_feat = self.borderConv(
            x=borders[:, 3: 5],
            emb=borders[:, :3],
            t=borders[:, 5:],
        )

        x = torch.cat(
            [ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs],
            dim=1,
        )

        outp = self.last_emb(x)

        return outp

class TrackerOffsetEmb(nn.Module):

    def __init__(
        self,
        device: torch.device,
        num_points: int=250,
        border_ic: int=6,
        outputD: int=64,
    ) -> None:

        super().__init__()

        self.point_feat = PoseNetFeatOffsetEmb(
            num_points=num_points,
            ic=3,
            border_ic=border_ic,
            output_dim=outputD,
        )

        self.num_points = num_points - num_points // 3
        self.embedding = LocationEmbedding(device)

    def forward(
        self,
        points: torch.Tensor,
        xyxys: torch.Tensor,
    ) -> torch.Tensor:

        embeds = self.embedding(xyxys)

        envs = points[:, self.num_points:]
        points = points[:, :self.num_points, :5]

        output = self.point_feat(
            inp=points.permute(0, 2, 1),
            borders=envs.permute(0, 2, 1),
            spatialEmbs=embeds,
        )

        return output
