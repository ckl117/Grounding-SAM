# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Sam model configuration"""

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["SAM_PRETRAINED_INIT_CONFIGURATION", "SamConfig", "SAM_PRETRAINED_RESOURCE_FILES_MAP"]

SAM_PRETRAINED_INIT_CONFIGURATION = {
    "SamVitH": {
        "modelname" : "SamVitH",
        "encoder_embed_dim" : 1280,
        "encoder_depth" : 32,
        "encoder_num_heads" : 16,
        "encoder_global_attn_indexes" : [7, 15, 23, 31],
        "input_type" : None
    },
    "SamVitL": {
        "modelname" : "SamVitL",
        "encoder_embed_dim" : 1024,
        "encoder_depth" : 24,
        "encoder_num_heads" : 16,
        "encoder_global_attn_indexes" : [5, 11, 17, 23],
        "input_type" : None
    },
    "SamVitB": {
        "modelname" : "SamVitB",
        "encoder_embed_dim" : 768,
        "encoder_depth" : 12,
        "encoder_num_heads" : 12,
        "encoder_global_attn_indexes" : [2, 5, 8, 11],
        "input_type" : None
    }
}

SAM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        'SamVitH': "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
        'SamVitL': "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
        'SamVitB': "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams"
    }
}


class SamConfig(PretrainedConfig):
  
    model_type = "Sam"
    pretrained_init_configuration = SAM_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        modelname = "Sam",
        encoder_embed_dim = 768,
        encoder_depth = 12,
        encoder_num_heads = 12,
        encoder_global_attn_indexes = [2, 5, 8, 11],
        input_type = None
    ):
        super().__init__()
        self.modelname = modelname
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_global_attn_indexes = encoder_global_attn_indexes
        self.input_type =  input_type
        self.pixel_mean = [123.675, 116.28, 103.53],
        self.pixel_std = [58.395, 57.12, 57.375]
