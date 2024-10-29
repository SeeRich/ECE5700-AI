# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Dict

import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    from mobile_sam.modeling import TinyViT


def make_tiny_vit() -> TinyViT:
    return TinyViT(
        img_size=1024,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )


class SamImageEncoder(nn.Module):
    """A wrapper around a TinyViT model to encode images into embeddings."""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.image_encoder = make_tiny_vit()

    def get_image_size(self) -> int:
        return self.image_encoder.img_size

    def forward(
        self,
        batched_input: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        return self.image_encoder(batched_input)
