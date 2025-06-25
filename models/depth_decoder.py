# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from models.layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.skip_connect = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_dec = np.array([32, 64, 128])
        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.skip_connect and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            if i == 0:
                self.convs[("dispconv")] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    # def forward(self, input_features):
    #     self.outputs = {}

    #     # decoder
    #     x = input_features[-1]
    #     for i in range(3, 0, -1):
    #         x = self.convs[("upconv", i-1, 0)](x)
    #         x = [upsample(x)]
    #         if self.use_skips and i > 1:
    #             x += [input_features[i - 1]]
    #         x = torch.cat(x, 1)
    #         x = self.convs[("upconv", i-1, 1)](x)
    #         if i-1 in self.scales:
    #             self.outputs[("disp", i-1)] = self.sigmoid(self.convs[("dispconv", i-1)](x))

        # return self.outputs
    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, input_dict):
        sz_in = input_dict[1].shape[3]

        x = input_dict[8]
        out = {8: x}

        if self.skip_connect:
            x = self.convs[("upconv", 2, 0)](x)
            x = [upsample(x)]
            x += [input_dict[4]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", 2, 1)](x)
            self.update_skip_dict(out, x, sz_in)
            x = self.convs[("upconv", 1, 0)](x)
            x = [upsample(x)]
            x += [input_dict[2]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", 1, 1)](x)
            self.update_skip_dict(out, x, sz_in)
            x = self.convs[("upconv", 0, 0)](x)
            x = [upsample(x)]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", 0, 1)](x)
            x = self.convs[("dispconv")](x)
            self.update_skip_dict(out, x, sz_in)
            out_depth = torch.sigmoid(out[1])
            out[1] = out_depth
        return out
