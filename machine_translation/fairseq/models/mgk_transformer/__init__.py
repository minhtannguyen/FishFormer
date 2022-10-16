# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .mgk_transformer_config import (
    MGKTransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .mgk_transformer_decoder import MGKTransformerDecoder, MGKTransformerDecoderBase, Linear
from .mgk_transformer_encoder import MGKTransformerEncoder, MGKTransformerEncoderBase
from .mgk_transformer_legacy import (
    MGKTransformerModel,
    mgk_base_architecture,
    mgk_tiny_architecture,
    mgk_transformer_iwslt_de_en,
    mgk_transformer_wmt_en_de,
    mgk_transformer_vaswani_wmt_en_de_big,
    mgk_transformer_vaswani_wmt_en_fr_big,
    mgk_transformer_wmt_en_de_big,
    mgk_transformer_wmt_en_de_big_t2t,
)
from .mgk_transformer_base import MGKTransformerModelBase, Embedding


__all__ = [
    "MGKTransformerModelBase",
    "MGKTransformerConfig",
    "MGKTransformerDecoder",
    "MGKTransformerDecoderBase",
    "MGKTransformerEncoder",
    "MGKTransformerEncoderBase",
    "MGKTransformerModel",
    "Embedding",
    "Linear",
    "mgk_base_architecture",
    "mgk_tiny_architecture",
    "mgk_transformer_iwslt_de_en",
    "mgk_transformer_wmt_en_de",
    "mgk_transformer_vaswani_wmt_en_de_big",
    "mgk_transformer_vaswani_wmt_en_fr_big",
    "mgk_transformer_wmt_en_de_big",
    "mgk_transformer_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
