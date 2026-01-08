# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class MultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        """
        :param x: [list] List of Tensors of different seq lengths
        :param masks: [list[list]] List of Tensors (out index: masks for given seq length, inner index: multimasks for that seq len)
        """
        if masks is None:
            return [self.backbone(xi) for xi in x]

        outs = [[] for _ in x]
        # -- This iterate through each sub batch
        for i, (xi, mi) in enumerate(zip(x, masks)):
            # -- This iterates through a list of masks
            for mij in mi:
                outs[i] += [self.backbone(xi, masks=mij)]
        # [sub_batch, num_masks, out]
        return outs


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, has_cls=False):
        """
        :param x: List[List[Tensor]] - Outer list is sub-batches (scales), inner list is views/masks.
        :param masks_x: List[List[Tensor]] - Corresponding masks for context.
        :param masks_y: List[List[Tensor]] - Corresponding masks for target.
        """
        outs = [[] for _ in x]
        # -- Iterate through sub batch
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            # -- Iterate through a list of masks
            for xij, mxij, myij in zip(xi, mxi, myi):
                outs[i] += [self.backbone(xij, mxij, myij, mask_index=i, has_cls=has_cls)]
        return outs
