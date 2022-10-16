# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .hdp_transformer_config import (
    HDPTransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .hdp_transformer_decoder import HDPTransformerDecoder, HDPTransformerDecoderBase, Linear
from .hdp_transformer_encoder import HDPTransformerEncoder, HDPTransformerEncoderBase

from .hdp_transformer_legacy import (
    HDPTransformerModel,
    hdp_base_architecture,
    hdp_tiny_architecture,
    hdp_transformer_iwslt_de_en_2head,
    hdp_transformer_iwslt_de_en_4head_2q,
    hdp_transformer_iwslt_de_en_4head_2k,
    hdp_transformer_iwslt_de_en_4head_2qk,
    hdp_transformer_iwslt_de_en_4head_2gqk,
    hdp_transformer_wmt_en_de,
    hdp_transformer_vaswani_wmt_en_de_big_4qk,
    hdp_transformer_vaswani_wmt_en_de_big_8qk,
    hdp_transformer_vaswani_wmt_en_fr_big,
    hdp_transformer_wmt_en_de_big,
    hdp_transformer_wmt_en_de_big_t2t,
)
from .hdp_transformer_base import HDPTransformerModelBase, Embedding


__all__ = [
    "HDPTransformerModelBase",
    "HDPTransformerConfig",
    "HDPTransformerDecoder",
    "HDPTransformerDecoderBase",
    "HDPTransformerEncoder",
    "HDPTransformerEncoderBase",
    "HDPTransformerModel",
    "Embedding",
    "Linear",
    "hdp_base_architecture",
    "hdp_tiny_architecture",
    "hdp_transformer_iwslt_de_en_2head",
    "hdp_transformer_iwslt_de_en_4head_2q",
    'hdp_transformer_iwslt_de_en_4head_2k',
    'hdp_transformer_iwslt_de_en_4head_2qk',
    'hdp_transformer_iwslt_de_en_4head_2gqk',
    "hdp_transformer_wmt_en_de",
    "hdp_transformer_vaswani_wmt_en_de_big_4qk",
    'hdp_transformer_vaswani_wmt_en_de_big_8qk',
    "hdp_transformer_vaswani_wmt_en_fr_big",
    "hdp_transformer_wmt_en_de_big",
    "hdp_transformer_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
