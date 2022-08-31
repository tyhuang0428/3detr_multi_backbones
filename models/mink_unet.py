try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')
    # blocks are used in the static part of MinkUNet
    BasicBlock, Bottleneck = None, None

import torch.nn as nn
import torch 
from third_party.pointnet2 import pointnet2_utils
import numpy as np


class MinkUNetBase(nn.Module):
    r"""Minkowski UNet backbone. See paper `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ and official
    `code <https://github.com/NVIDIA/MinkowskiEngine/
    blob/master/examples/minkunet.py>`_  for more details.
    This backbone is the reimplementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.
    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        planes (str, optional): Planes unet config.
            Defaults to None.
        D (int): Dimension which will be applied in.
            Defaults to 3.
    """

    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    arch_settings = {
        14: {
            'block': BasicBlock,
            'layers': (1, 1, 1, 1, 1, 1, 1, 1),
            'planes': {
                'A': (32, 64, 128, 256, 128, 128, 96, 96),
                'B': (32, 64, 128, 256, 128, 128, 128, 128),
                'C': (32, 64, 128, 256, 192, 192, 128, 128),
                'D': (32, 64, 128, 256, 384, 384, 384, 384),
            },
        },
        18: {
            'block': BasicBlock,
            'layers': (2, 2, 2, 2, 2, 2, 2, 2),
            'planes': {
                'A': (32, 64, 128, 256, 128, 128, 96, 96),
                'B': (32, 64, 128, 256, 128, 128, 128, 128),
                'D': (32, 64, 128, 256, 384, 384, 384, 384),
            },
        },
        34: {
            'block': BasicBlock,
            'layers': (2, 3, 4, 6, 2, 2, 2, 2),
            'planes': {
                'A': (32, 64, 128, 256, 256, 128, 64, 64),
                'B': (32, 64, 128, 256, 256, 128, 64, 32),
                'C': (32, 64, 128, 256, 256, 128, 96, 96),
            },
        },
        50: {
            'block': Bottleneck,
            'layers': (2, 3, 4, 6, 2, 2, 2, 2),
        },
        101: {
            'block': Bottleneck,
            'layers': (2, 3, 4, 23, 2, 2, 2, 2),
        },
    }

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, depth, in_channels, npoints, ckpt, out_channel=256, planes=None, D=3):
        super(MinkUNetBase, self).__init__()
        self.in_channels = in_channels
        self.npoints = npoints
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for unet.')

        if 'planes' not in self.arch_settings[
                depth] or planes not in self.arch_settings[depth]['planes']:
            self.planes = self.PLANES
        else:
            self.planes = self.arch_settings[depth]['planes'][planes]

        self.block = self.arch_settings[depth]['block']
        self.layers = self.arch_settings[depth]['layers']
        self.D = D

        self.final = ME.MinkowskiConvolution(
            96,
            out_channel,
            kernel_size=1,
            bias=True,
            dimension=D,
        )
        self.network_initialization(self.in_channels, D)
        self.weight_initialization(ckpt)

    def weight_initialization(self, ckpt):
        if ckpt is not None:
            print('loading ckpt from: %s' % ckpt)
            base_ckpt = torch.load(ckpt)['state_dict']
            for k in list(base_ckpt.keys()):
                if k.startswith('final.') :
                    del base_ckpt[k]    
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            print(incompatible.missing_keys)
            print(incompatible.unexpected_keys)
        else:
            for m in self.modules():
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel,
                        mode='fan_out',
                        nonlinearity='relu',
                    )

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)

    def network_initialization(self, in_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.block, self.planes[0],
                                       self.layers[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.block, self.planes[1],
                                       self.layers[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.block, self.planes[2],
                                       self.layers[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.block, self.planes[3],
                                       self.layers[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.planes[4],
            kernel_size=2,
            stride=2,
            dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.planes[4])

        self.inplanes = self.planes[4] + self.planes[2] * self.block.expansion
        self.block5 = self._make_layer(self.block, self.planes[4],
                                       self.layers[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.planes[5],
            kernel_size=2,
            stride=2,
            dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.planes[5])

        self.inplanes = self.planes[5] + self.planes[1] * self.block.expansion
        self.block6 = self._make_layer(self.block, self.planes[5],
                                       self.layers[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.planes[6],
            kernel_size=2,
            stride=2,
            dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.planes[6])

        self.inplanes = self.planes[6] + self.planes[0] * self.block.expansion
        self.block7 = self._make_layer(self.block, self.planes[6],
                                       self.layers[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.planes[7],
            kernel_size=2,
            stride=2,
            dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.planes[7])

        self.inplanes = self.planes[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.block, self.planes[7],
                                       self.layers[7])

        self.relu = ME.MinkowskiReLU(inplace=True)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    dimension=self.D))

        return nn.Sequential(*layers)

    def forward(self, xyz):
        feats = []
        points = []
        list_of_coords = []
        for i in range(xyz.shape[0]):
            _, idx = ME.utils.sparse_quantize(xyz[i] / 0.02, return_index=True)
            
            cur_xyz = xyz[i]
            points.append(torch.floor(cur_xyz[idx] / 0.02))
            list_of_coords.append(cur_xyz[idx])
            feats.append(torch.ones_like(cur_xyz[idx], device=xyz.device))

            # points.append(xyz[i])
            # feats.append(torch.ones_like(xyz[i], device=xyz.device))
        points, feats = ME.utils.sparse_collate(points, feats)

        x = ME.SparseTensor(features=feats, coordinates=points, device=xyz.device)
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)
        out = self.final(out)
        _, list_of_features = out.decomposed_coordinates_and_features
        out_coords = []
        out_features = []
        for i in range(len(list_of_coords)):
            coords = list_of_coords[i].unsqueeze(dim=0) #* 0.02
            feats = list_of_features[i].unsqueeze(dim=0)
            idx = pointnet2_utils.furthest_point_sample(coords, self.npoints)
            out_coords.append(pointnet2_utils.gather_operation(coords.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous())
            out_features.append(pointnet2_utils.gather_operation(feats.transpose(1, 2).contiguous(), idx))
        return torch.cat(out_coords), torch.cat(out_features).permute(2, 0, 1), None
    # def forward(self, xyz):
    #     # xyz /= 0.02
    #     # feats = []
    #     coords = []
    #     inds = []
    #     for i in range(xyz.shape[0]):
    #         cur_xyz = torch.floor(xyz[i] / 0.02)
    #         _, idx = ME.utils.sparse_quantize(cur_xyz, return_index=True)
    #         coords.append(cur_xyz)
    #         # feats.append(torch.ones_like(cur_xyz[idx], device=xyz.device))
    #         inds.append(np.array(idx, dtype=np.int32))

    #     batch_ids = np.array([b for b, v in enumerate(coords) for _ in range(v.shape[0])])
    #     voxel_ids = np.concatenate(inds, 0)
    #     coords

    #     points, feats = ME.utils.sparse_collate(points, feats)

    #     x = ME.SparseTensor(features=feats, coordinates=points, device=xyz.device)
    #     out = self.conv0p1s1(x)
    #     out = self.bn0(out)
    #     out_p1 = self.relu(out)

    #     out = self.conv1p1s2(out_p1)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #     out_b1p2 = self.block1(out)

    #     out = self.conv2p2s2(out_b1p2)
    #     out = self.bn2(out)
    #     out = self.relu(out)
    #     out_b2p4 = self.block2(out)

    #     out = self.conv3p4s2(out_b2p4)
    #     out = self.bn3(out)
    #     out = self.relu(out)
    #     out_b3p8 = self.block3(out)

    #     # tensor_stride=16
    #     out = self.conv4p8s2(out_b3p8)
    #     out = self.bn4(out)
    #     out = self.relu(out)
    #     out = self.block4(out)

    #     # tensor_stride=8
    #     out = self.convtr4p16s2(out)
    #     out = self.bntr4(out)
    #     out = self.relu(out)

    #     out = ME.cat(out, out_b3p8)
    #     out = self.block5(out)

    #     # tensor_stride=4
    #     out = self.convtr5p8s2(out)
    #     out = self.bntr5(out)
    #     out = self.relu(out)

    #     out = ME.cat(out, out_b2p4)
    #     out = self.block6(out)

    #     # tensor_stride=2
    #     out = self.convtr6p4s2(out)
    #     out = self.bntr6(out)
    #     out = self.relu(out)

    #     out = ME.cat(out, out_b1p2)
    #     out = self.block7(out)

    #     # tensor_stride=1
    #     out = self.convtr7p2s2(out)
    #     out = self.bntr7(out)
    #     out = self.relu(out)

    #     out = ME.cat(out, out_p1)
    #     out = self.block8(out)
    #     out = self.final(out)
    #     list_of_coords, list_of_features = out.decomposed_coordinates_and_features
    #     out_coords = []
    #     out_features = []
    #     for i in range(len(list_of_coords)):
    #         coords = list_of_coords[i].unsqueeze(dim=0) * 0.02
    #         feats = list_of_features[i].unsqueeze(dim=0)
    #         idx = pointnet2_utils.furthest_point_sample(coords, self.npoints)
    #         out_coords.append(pointnet2_utils.gather_operation(coords.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous())
    #         out_features.append(pointnet2_utils.gather_operation(feats.transpose(1, 2).contiguous(), idx))
    #     return torch.cat(out_coords), torch.cat(out_features).permute(2, 0, 1), None