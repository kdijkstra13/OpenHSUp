# Copyright (C) 2020 Klaas Dijkstra
#
# This file is part of OpenHSUp.
#
# OpenHSUp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# OpenHSUp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with OpenHSUp.  If not, see <https://www.gnu.org/licenses/>.

from torch import nn
import torch.nn.functional as F
import torch
from collections import OrderedDict


class HSConvert(nn.Module):
    def __init__(self, mosaic_size: int):
        super(HSConvert, self).__init__()
        assert mosaic_size > 1, "This is not a mosaic image"

        self.conv2d = nn.Conv2d(1, mosaic_size**2, kernel_size=mosaic_size, padding=0, stride=mosaic_size, bias=False)
        self.mosaic_size = mosaic_size
        self.init_weights()

    def init_weights(self) -> 'HSConvert':
        self.conv2d.weight.data[:] = 0
        for y in range(self.conv2d.weight.shape[2]):
            for x in range(self.conv2d.weight.shape[3]):
                c = y * self.mosaic_size + x
                self.conv2d.weight.data[c, 0, y, x] = 1
        return self

    def forward(self, x):
        return self.conv2d(x)


class HSUp2(nn.Module):
    def __init__(self, num_channels: int, num_filters: int, filter_size: int):
        super(HSUp2, self).__init__()
        self.scale = 2
        self.filter_size = max(2 * self.scale - 1, filter_size)  # Smaller makes not sense.
        self.num_filters = num_filters
        self.num_channels = num_channels

        stride = self.scale
        out_pad = filter_size % 2
        pad = (filter_size + out_pad - 1 * stride) // 2

        h = 11
        w = 11
        h_ = (h - 1) * stride - 2 * pad + filter_size + out_pad
        w_ = (w - 1) * stride - 2 * pad + filter_size + out_pad
        assert h_ == h * self.scale or w_ == h * self.scale, "Invalid padding [{} x {}] * {} != [{} x {}]".format(h, w, self.scale, h_, w_)

        self.conv2dt = nn.ConvTranspose2d(num_channels, num_filters, kernel_size=filter_size, stride=2, padding=pad, output_padding=out_pad)

    def forward(self, x):
        return F.relu(self.conv2dt(x))

    def init_weights(self):
        if self.filter_size != 2 * self.scale - 1:
            print("Warning: filter size should typically be {}".format(2 * self.scale - 1))
        if self.num_filters != self.num_channels:
            print("Warning: num_filters should typically be {}".format(min(self.num_channels, self.num_filters)))
        nn.init.constant_(self.conv2dt.bias, 0)
        nn.init.constant_(self.conv2dt.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.conv2dt.stride)
        for i in range(min(self.num_channels, self.num_filters)):
            if self.conv2dt.groups == 1:
                j = i
            else:
                j = 0
            self.conv2dt.weight.data[i, j][:bilinear_kernel.shape[0], :bilinear_kernel.shape[1]] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        num_dims = len(stride)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel channel by channel
        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            # e.g. with stride = 4
            # delta = [-3, -2, -1, 0, 1, 2, 3]
            # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            delta = torch.arange(1 - channel_stride, channel_stride).type(bilinear_kernel.type())
            channel_filter = (1 - torch.abs(delta / channel_stride))
            # Apply the channel filter to the current channel
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel


class HSUp(nn.Module):
    def __init__(self, mosaic_size: int = 4, num_filters: int = 128, filter_size: int = 8):
        super(HSUp, self).__init__()
        self.mosaic_size = mosaic_size
        self.num_channels = mosaic_size ** 2
        self.num_filters = num_filters
        self.filter_size = filter_size

        if mosaic_size == 4: #4 x 4
            self.network = nn.Sequential(OrderedDict([
                ("convert", HSConvert(self.mosaic_size)),
                ("hsup1", HSUp2(self.num_channels, num_filters, filter_size)),
                ("hsup2", HSUp2(self.num_filters, self.num_channels, filter_size))]))
        else:
            raise Exception("Invalid mosaic size, add a new one by changing the model {}".format(mosaic_size))

    def init_weights(self):
        # Init with a bilinear kernel
        self.network._modules["hsup1"].init_weights()
        self.network._modules["hsup2"].init_weights()

    def clone(self):
        m = HSUp(mosaic_size=self.mosaic_size, num_filters=self.num_filters, filter_size=self.filter_size)
        m.load_state_dict(self.state_dict())
        return m

    def forward(self, x):
        x = self.network(x)
        return x
