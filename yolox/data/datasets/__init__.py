#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .dior import DIORDetection
from .dota import DOTADetection
from .dota_obb import DOTAOBBDetection
from .mosaicdetection import MosaicDetection
from .mosaicdetection_obb import MosaicDetectionOBB
from .voc import VOCDetection
