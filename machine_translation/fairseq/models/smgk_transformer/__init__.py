# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .smgk_transformer_config import (
    SMGKTransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .smgk_transformer_decoder import SMGKTransformerDecoder, SMGKTransformerDecoderBase, Linear
from .smgk_transformer_encoder import SMGKTransformerEncoder, SMGKTransformerEncoderBase
from .smgk_transformer_legacy import (
    SMGKTransformerModel,
    smgk_base_architecture,
    smgk_tiny_architecture,
    smgk_transformer_iwslt_de_en_2head,
    smgk_transformer_iwslt_de_en_4head,
    smgk_transformer_wmt_en_de,
    smgk_transformer_vaswani_wmt_en_de_big,
    smgk_transformer_vaswani_wmt_en_fr_big,
    smgk_transformer_wmt_en_de_big,
    smgk_transformer_wmt_en_de_big_t2t,
)
from .smgk_transformer_base import SMGKTransformerModelBase, Embedding


__all__ = [
    "SMGKTransformerModelBase",
    "SMGKTransformerConfig",
    "SMGKTransformerDecoder",
    "SMGKTransformerDecoderBase",
    "SMGKTransformerEncoder",
    "SMGKTransformerEncoderBase",
    "SMGKTransformerModel",
    "Embedding",
    "Linear",
    "smgk_base_architecture",
    "smgk_tiny_architecture",
    "smgk_transformer_iwslt_de_en_2head",
    "smgk_transformer_iwslt_de_en_4head",
    "smgk_transformer_wmt_en_de",
    "smgk_transformer_vaswani_wmt_en_de_big",
    "smgk_transformer_vaswani_wmt_en_fr_big",
    "smgk_transformer_wmt_en_de_big",
    "smgk_transformer_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
