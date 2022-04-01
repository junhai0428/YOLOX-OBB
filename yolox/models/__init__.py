#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .KLD_loss import compute_kld_loss, KLDloss
from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_head_obb_kld import YOLOXHeadOBB_KLD
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
from .yolox_obb_kld import YOLOXOBB_KLD
