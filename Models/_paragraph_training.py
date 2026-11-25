"""
PARAGraph Training Mixin

Training control: freeze/unfreeze backbone, finetune setup, parameter statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .paragraph import PARAGraph


class PARAGraphTrainingMixin:
    """Mixin: training control (freeze/unfreeze, finetune, parameter statistics)."""

    def freeze_backbone(self):
        """
        Freeze the entire backbone (all OpticUNet and LesionUNet parameters).
        """
        for param in self.optic_unet.parameters():
            param.requires_grad = False
        if self.lesion_unet is not None:
            for param in self.lesion_unet.parameters():
                param.requires_grad = False
        print("Backbone frozen (OpticUNet + LesionUNet)")

    def unfreeze_backbone(self):
        """
        Unfreeze the entire backbone.
        """
        for param in self.optic_unet.parameters():
            param.requires_grad = True
        if self.lesion_unet is not None:
            for param in self.lesion_unet.parameters():
                param.requires_grad = True
        print("Backbone unfrozen (OpticUNet + LesionUNet)")

    def freeze_backbone_encoder(self):
        """
        Freeze the backbone encoder (keep the decoder trainable).
        Supports both ResNet50 and Swin Transformer encoders.
        """
        self.optic_unet._freeze_encoder()

        if self.lesion_unet is not None:
            self.lesion_unet._freeze_encoder()

        print("Backbone encoder frozen")

    def unfreeze_backbone_decoder(
        self,
        optic_od: bool = True,
        optic_landmark: bool = True,
        lesion: bool = True,
        stages: Optional[List[int]] = None
    ):
        """
        Unfreeze the backbone decoder for finetuning.

        Args:
            optic_od: whether to unfreeze the OpticUNet OD decoder
            optic_landmark: whether to unfreeze the OpticUNet Landmark decoder
            lesion: whether to unfreeze the LesionUNet decoder
            stages: decoder stages to unfreeze (1-4); None unfreezes all.
                    e.g. [3, 4] unfreezes only the stage3 and stage4 decoders
        """
        if stages is None:
            stages = [1, 2, 3, 4]

        # OpticUNet OD decoder
        if optic_od:
            od_decoders = {
                4: self.optic_unet.od_decoder_stage4,
                3: self.optic_unet.od_decoder_stage3,
                2: self.optic_unet.od_decoder_stage2,
                1: self.optic_unet.od_decoder_stage1,
            }
            for stage in stages:
                if stage in od_decoders:
                    for param in od_decoders[stage].parameters():
                        param.requires_grad = True

            # final upsample and segmentation head
            if 1 in stages:
                for param in self.optic_unet.od_final_upsample.parameters():
                    param.requires_grad = True
                for param in self.optic_unet.od_head.parameters():
                    param.requires_grad = True

            print(f"OpticUNet OD decoder unfrozen (stages: {stages})")

        # OpticUNet Landmark decoder
        if optic_landmark and self.optic_unet.predict_fovea:
            landmark_decoders = {
                4: self.optic_unet.fovea_decoder_stage4,
                3: self.optic_unet.fovea_decoder_stage3,
                2: self.optic_unet.fovea_decoder_stage2,
                1: self.optic_unet.fovea_decoder_stage1,
            }
            for stage in stages:
                if stage in landmark_decoders:
                    for param in landmark_decoders[stage].parameters():
                        param.requires_grad = True

            # final upsample and heatmap head
            if 1 in stages:
                for param in self.optic_unet.fovea_final_upsample.parameters():
                    param.requires_grad = True
                for param in self.optic_unet.fovea_head.parameters():
                    param.requires_grad = True

            print(f"OpticUNet Landmark decoder unfrozen (stages: {stages})")

        # LesionUNet decoder
        if lesion and self.lesion_unet is not None:
            lesion_decoders = {
                4: self.lesion_unet.decoder_stage4,
                3: self.lesion_unet.decoder_stage3,
                2: self.lesion_unet.decoder_stage2,
                1: self.lesion_unet.decoder_stage1,
            }
            for stage in stages:
                if stage in lesion_decoders:
                    for param in lesion_decoders[stage].parameters():
                        param.requires_grad = True

            # final upsample and segmentation head
            if 1 in stages:
                for param in self.lesion_unet.final_upsample.parameters():
                    param.requires_grad = True
                for param in self.lesion_unet.segmentation_head.parameters():
                    param.requires_grad = True

            print(f"LesionUNet decoder unfrozen (stages: {stages})")

    def setup_finetune_decoder(
        self,
        lr_decoder: float = 1e-4,
        lr_other: float = 1e-4,
        stages: Optional[List[int]] = None,
        optic_od: bool = True,
        optic_landmark: bool = True,
        lesion: bool = True
    ) -> List[dict]:
        """
        Build decoder-finetune parameter groups for creating an optimizer.

        Usage:
            param_groups = model.setup_finetune_decoder(lr_decoder=1e-5, lr_other=1e-4)
            optimizer = torch.optim.AdamW(param_groups)

        Args:
            lr_decoder: learning rate for the decoder (usually smaller)
            lr_other: learning rate for the rest (GNN, classifier, etc.)
            stages: decoder stages to unfreeze
            optic_od: whether to unfreeze the OpticUNet OD decoder
            optic_landmark: whether to unfreeze the OpticUNet Landmark decoder
            lesion: whether to unfreeze the LesionUNet decoder

        Returns:
            param_groups: list of parameter groups for the optimizer
        """
        # first freeze the entire backbone
        self.freeze_backbone()

        # unfreeze the specified decoders
        self.unfreeze_backbone_decoder(
            optic_od=optic_od,
            optic_landmark=optic_landmark,
            lesion=lesion,
            stages=stages
        )

        # collect decoder parameters
        decoder_params = []

        if optic_od:
            decoder_params.extend(list(self.optic_unet.od_decoder_stage4.parameters()))
            decoder_params.extend(list(self.optic_unet.od_decoder_stage3.parameters()))
            decoder_params.extend(list(self.optic_unet.od_decoder_stage2.parameters()))
            decoder_params.extend(list(self.optic_unet.od_decoder_stage1.parameters()))
            decoder_params.extend(list(self.optic_unet.od_final_upsample.parameters()))
            decoder_params.extend(list(self.optic_unet.od_head.parameters()))

        if optic_landmark and self.optic_unet.predict_fovea:
            decoder_params.extend(list(self.optic_unet.fovea_decoder_stage4.parameters()))
            decoder_params.extend(list(self.optic_unet.fovea_decoder_stage3.parameters()))
            decoder_params.extend(list(self.optic_unet.fovea_decoder_stage2.parameters()))
            decoder_params.extend(list(self.optic_unet.fovea_decoder_stage1.parameters()))
            decoder_params.extend(list(self.optic_unet.fovea_final_upsample.parameters()))
            decoder_params.extend(list(self.optic_unet.fovea_head.parameters()))

        if lesion and self.lesion_unet is not None:
            decoder_params.extend(list(self.lesion_unet.decoder_stage4.parameters()))
            decoder_params.extend(list(self.lesion_unet.decoder_stage3.parameters()))
            decoder_params.extend(list(self.lesion_unet.decoder_stage2.parameters()))
            decoder_params.extend(list(self.lesion_unet.decoder_stage1.parameters()))
            decoder_params.extend(list(self.lesion_unet.final_upsample.parameters()))
            decoder_params.extend(list(self.lesion_unet.segmentation_head.parameters()))

        # keep only requires_grad=True parameters
        decoder_params = [p for p in decoder_params if p.requires_grad]
        decoder_param_ids = set(id(p) for p in decoder_params)

        # other trainable parameters (GNN, classifier, projection layers, etc.)
        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in decoder_param_ids
        ]

        param_groups = [
            {'params': decoder_params, 'lr': lr_decoder, 'name': 'decoder'},
            {'params': other_params, 'lr': lr_other, 'name': 'other'}
        ]

        # count parameters
        num_decoder = sum(p.numel() for p in decoder_params)
        num_other = sum(p.numel() for p in other_params)
        print(f"Finetune setup: decoder params={num_decoder:,}, other params={num_other:,}")
        print(f"Learning rates: decoder={lr_decoder}, other={lr_other}")

        return param_groups

    def get_decoder_params(
        self,
        optic_od: bool = False,
        optic_landmark: bool = False,
        lesion: bool = False
    ) -> List[nn.Parameter]:
        """
        Get the parameter list for the specified decoders.

        Args:
            optic_od: whether to include OD decoder parameters
            optic_landmark: whether to include Landmark decoder parameters
            lesion: whether to include Lesion decoder parameters

        Returns:
            params: list of parameters
        """
        params = []

        if optic_od:
            params.extend(list(self.optic_unet.od_decoder_stage4.parameters()))
            params.extend(list(self.optic_unet.od_decoder_stage3.parameters()))
            params.extend(list(self.optic_unet.od_decoder_stage2.parameters()))
            params.extend(list(self.optic_unet.od_decoder_stage1.parameters()))
            params.extend(list(self.optic_unet.od_final_upsample.parameters()))
            params.extend(list(self.optic_unet.od_head.parameters()))

        if optic_landmark and self.optic_unet.predict_fovea:
            params.extend(list(self.optic_unet.fovea_decoder_stage4.parameters()))
            params.extend(list(self.optic_unet.fovea_decoder_stage3.parameters()))
            params.extend(list(self.optic_unet.fovea_decoder_stage2.parameters()))
            params.extend(list(self.optic_unet.fovea_decoder_stage1.parameters()))
            params.extend(list(self.optic_unet.fovea_final_upsample.parameters()))
            params.extend(list(self.optic_unet.fovea_head.parameters()))

        if lesion and self.lesion_unet is not None:
            params.extend(list(self.lesion_unet.decoder_stage4.parameters()))
            params.extend(list(self.lesion_unet.decoder_stage3.parameters()))
            params.extend(list(self.lesion_unet.decoder_stage2.parameters()))
            params.extend(list(self.lesion_unet.decoder_stage1.parameters()))
            params.extend(list(self.lesion_unet.final_upsample.parameters()))
            params.extend(list(self.lesion_unet.segmentation_head.parameters()))

        # return only requires_grad=True parameters
        return [p for p in params if p.requires_grad]

    def get_trainable_params_info(self) -> dict:
        """
        Get statistics about currently trainable parameters.

        Returns:
            info: dict with trainable parameter counts per module
        """
        info = {
            'total': 0,
            'trainable': 0,
            'frozen': 0,
            'modules': {}
        }

        # per-module statistics (OpticUNet: ResNet-50; LesionUNet: ResNet-50 or Swin)
        module_groups = {}

        # Optic encoder
        module_groups['optic_encoder'] = [
            self.optic_unet.encoder_stem,
            self.optic_unet.encoder_stage1,
            self.optic_unet.encoder_stage2,
            self.optic_unet.encoder_stage3,
            self.optic_unet.encoder_stage4,
        ]

        # Optic OD decoder
        module_groups['optic_od_decoder'] = [
            self.optic_unet.od_decoder_stage4,
            self.optic_unet.od_decoder_stage3,
            self.optic_unet.od_decoder_stage2,
            self.optic_unet.od_decoder_stage1,
            self.optic_unet.od_final_upsample,
            self.optic_unet.od_head,
        ]

        if self.optic_unet.predict_fovea:
            module_groups['optic_landmark_decoder'] = [
                self.optic_unet.fovea_decoder_stage4,
                self.optic_unet.fovea_decoder_stage3,
                self.optic_unet.fovea_decoder_stage2,
                self.optic_unet.fovea_decoder_stage1,
                self.optic_unet.fovea_final_upsample,
                self.optic_unet.fovea_head,
            ]

        if self.lesion_unet is not None:
            # Lesion encoder
            if hasattr(self.lesion_unet, 'swin_encoder'):
                module_groups['lesion_encoder'] = [self.lesion_unet.swin_encoder]
            else:
                module_groups['lesion_encoder'] = [
                    self.lesion_unet.encoder_stem,
                    self.lesion_unet.encoder_stage1,
                    self.lesion_unet.encoder_stage2,
                    self.lesion_unet.encoder_stage3,
                    self.lesion_unet.encoder_stage4,
                ]
            module_groups['lesion_decoder'] = [
                self.lesion_unet.decoder_stage4,
                self.lesion_unet.decoder_stage3,
                self.lesion_unet.decoder_stage2,
                self.lesion_unet.decoder_stage1,
                self.lesion_unet.final_upsample,
                self.lesion_unet.segmentation_head,
            ]

        for group_name, modules in module_groups.items():
            trainable = 0
            total = 0
            for module in modules:
                for p in module.parameters():
                    total += p.numel()
                    if p.requires_grad:
                        trainable += p.numel()
            info['modules'][group_name] = {
                'total': total,
                'trainable': trainable,
                'frozen': total - trainable
            }

        # count PARAGraph's own parameters
        paragraph_params = []
        backbone_param_ids = set()
        for p in self.optic_unet.parameters():
            backbone_param_ids.add(id(p))
        if self.lesion_unet is not None:
            for p in self.lesion_unet.parameters():
                backbone_param_ids.add(id(p))

        for p in self.parameters():
            if id(p) not in backbone_param_ids:
                paragraph_params.append(p)

        pg_trainable = sum(p.numel() for p in paragraph_params if p.requires_grad)
        pg_total = sum(p.numel() for p in paragraph_params)
        info['modules']['paragraph'] = {
            'total': pg_total,
            'trainable': pg_trainable,
            'frozen': pg_total - pg_trainable
        }

        # totals
        info['total'] = sum(p.numel() for p in self.parameters())
        info['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info['frozen'] = info['total'] - info['trainable']

        return info

    def print_trainable_params(self):
        """
        Print statistics about trainable parameters.
        """
        info = self.get_trainable_params_info()

        print("\n" + "=" * 60)
        print("Trainable Parameters Summary")
        print("=" * 60)

        for name, stats in info['modules'].items():
            status = "\u2713" if stats['trainable'] > 0 else "\u2717"
            print(f"{status} {name:25s}: {stats['trainable']:>10,} / {stats['total']:>10,}")

        print("-" * 60)
        print(f"{'Total':25s}: {info['trainable']:>10,} / {info['total']:>10,}")
        print(f"{'Frozen':25s}: {info['frozen']:>10,}")
        print("=" * 60 + "\n")
