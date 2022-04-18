#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):

    def __init__(
        self,
        ninput: int,
        noutput: int
    ) -> None:

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=ninput,
            out_channels=noutput-ninput,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            bias=True,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(
            num_features=noutput,
            eps=1e-3,
        )

    def forward(
        self,
        input: torch.tensor,
    ) -> torch.tensor:

        output = torch.cat(
            tensors=[self.conv(input), self.pool(input)],
            dim=1,
        )
        output = self.bn(output)

        return F.relu(output)

class non_bottleneck_1d(nn.Module):

    def __init__(
        self,
        chann: int,
        dropprob: int,
        dilated: int,
    ) -> None:

        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            in_channels=chann,
            out_channels=chann,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            bias=True,
        )

        self.conv1x3_1 = nn.Conv2d(
            in_channels=chann,
            out_channels=chann,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            bias=True,
        )

        self.bn1 = nn.BatchNorm2d(
            num_features=chann,
            eps=1e-03,
        )

        self.conv3x1_2 = nn.Conv2d(
            in_channels=chann,
            out_channels=chann,
            kernel_size=(3, 1),
            stride=1,
            padding=(dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            in_channels=chann,
            out_channels=chann,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(
            num_features=chann,
            eps=1e-03,
        )

        self.dropout = nn.Dropout2d(
            p=dropprob,
        )

    def forward(
        self,
        input: torch.tensor,
    ) -> torch.tensor:

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)


class Encoder(nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_channel: int=3,
    ) -> None:

        super().__init__()

        self.initial_block = DownsamplerBlock(input_channel, 16)

        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            in_channels=128,
            out_channels=num_classes,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        input: torch.tensor,
        predict: bool=False,
        feat: bool=False,
    ) -> torch.tensor:

        output = self.initial_block(input)
        feats = output

        for layer in self.layers:
            output = layer(output)

        if predict:
            att = self.output_conv(output)
            return output, att

        if feat:
            return output, feats

        return output

class UpsamplerBlock(nn.Module):

    def __init__(
        self,
        ninput: int,
        noutput: int,
    ) -> None:

        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels=ninput,
            out_channels=noutput,
            kernel_size=3,
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
            bias=True,
        )
        self.bn = nn.BatchNorm2d(
            num_features=noutput,
            eps=1e-3,
        )

    def forward(
        self,
        input: torch.tensor
    ) -> torch.tensor:

        output = self.conv(input)
        output = self.bn(output)

        return F.relu(output)

class Decoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ) -> None:

        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            output_padding=(0, 0), 
            bias=True,
        )

    def forward(
        self,
        input: torch.tensor,
    ) -> torch.tensor:

        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class Net(nn.Module):

    def __init__(
        self,
        num_classes: int,
        encoder=None,
    ) -> None:

        super().__init__()

        if encoder is None:
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder

        self.decoder = Decoder(num_classes)

    def forward(
        self,
        input: torch.tensor,
        only_encode: bool=False,
    ) -> torch.tensor:
    
        if only_encode:
            output = self.encoder(
                input=input,
                predict=True,
            )
            return output
        else:
            output = self.encoder(
                input=input,
                predict=False,
            )

            return self.decoder.forward(output)
