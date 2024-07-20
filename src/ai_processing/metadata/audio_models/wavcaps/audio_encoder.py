#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
from abc import ABC

import torch
import yaml
from transformers.modeling_outputs import BaseModelOutput
from app.ai_processing.metadata.audio_models.wavcaps.htsat import HTSAT_Swin_Transformer
from transformers import PreTrainedModel
from app.ai_processing.metadata.audio_models.wavcaps.audio_encoder_config import AudioEncoderConfig
import os
from config import Config
from app.utils import utils


class AudioEncoderModel(PreTrainedModel):
    config_class = AudioEncoderConfig

    def __init__(self, config):
        super(AudioEncoderModel, self).__init__(config)

        if config.model_arch == "transformer":
            self.audio_enc = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=527,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=config
            )
            if config.pretrained:
                ckpt_path = f"{Config.ROOT_FOLDER}/app/ai_processing/metadata/audio_models/wavcaps/pretrained_models/audio_encoder/HTSAT.ckpt"
                if os.path.exists(ckpt_path):
                    pass
                else:
                    print("Downloading HTSAT.ckpt from S3 bucket")
                    utils.download_from_s3(Config.ai_models_weights_s3, "HTSAT.ckpt", ckpt_path)                
                
                audio_ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                for key in list(audio_ckpt.keys()):
                    if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                        and 'logmel_extractor' not in key):
                        v = audio_ckpt.pop(key)
                        audio_ckpt[key[10:]] = v
                self.audio_enc.load_state_dict(audio_ckpt, strict=False)
                # param_names = [n for n, p in self.audio_enc.named_parameters()]
                # for n in param_names:
                #     print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
            self.audio_width = 768
        else:
            raise NotImplementedError('No such audio encoder network.')

        if config.freeze:
            for name, param in self.audio_enc.named_parameters():
                if "fc1" not in name:
                    param.requires_grad = False

    def forward(self, input_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
                ):
        audio_embeds = self.audio_enc(input_ids)
        if not return_dict:
            return (audio_embeds, )
        return BaseModelOutput(audio_embeds, None, None)

