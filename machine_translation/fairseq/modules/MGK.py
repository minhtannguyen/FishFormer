import torch
import math
from gpytorch.kernels.kernel import Distance
from torch.nn.functional import dropout, linear, pad
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
import warnings

from typing import Dict, Optional, Tuple

import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter


def rbf2keys_attention(q,k1,k2,v, pi, scaling,dist, num_head,attn_mask= None,dropout_p =0.0):

    B, Nt, E = q.shape
    Ns = k1.shape[1]
    # q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    
    QK1_distance = (-scaling/2)*dist._sq_dist(q, k1, postprocess = False)
    QK2_distance = (-3*scaling/2)*dist._sq_dist(q, k2, postprocess = False)
    if attn_mask is not None:
        QK1_distance += attn_mask
        QK2_distance += attn_mask

    #bsz, num_heads, tgt_len, src_len
    # attn = torch.exp(QK1_distance)*torch.clamp(torch.abs(pi[0]), min = 1e-6, max = 2.) + torch.exp(QK2_distance)*torch.clamp(torch.abs(pi[1]), min = 1e-6, max = 2.)

    attn = torch.exp(QK1_distance).view(B//num_head, -1, Nt, Ns)*torch.clamp(torch.abs(pi[0][None, :, None, None]), min = 1e-6, max = 2.) + torch.exp(QK2_distance).view(B//num_head, -1, Nt, Ns)*torch.clamp(torch.abs(pi[1][None, :, None, None]), min = 1e-6, max = 2.)
    attn = attn.view(-1, Nt, Ns)
    attn = attn/(attn.sum(dim = -1, keepdim = True) + 1e-6)
    
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn

@with_incremental_state
class MGKAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        head_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.dist = Distance()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = int(num_heads)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        #######FOR MGK 
        self.head_dim = int(head_dim)
        # self.pi = (torch.ones(2, self.num_heads)/2.).cuda()
        self.pi = Parameter(torch.ones(2,self.num_heads)/2., requires_grad = True)
        self.dist = Distance()
        # assert (
        #     self.head_dim * num_heads == self.embed_dim
        # ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        # print(self.head_dim*self.num_heads)
        # import pdb; pdb.set_trace()
        self.k_proj1 = quant_noise(
            nn.Linear(self.kdim, self.num_heads*self.head_dim, bias=bias), q_noise, qn_block_size
        )

        self.k_proj2 = quant_noise(
            nn.Linear(self.kdim, self.num_heads*self.head_dim, bias=bias), q_noise, qn_block_size
        )

        self.v_proj = quant_noise(
            nn.Linear(self.vdim, self.num_heads*self.head_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, self.num_heads*self.head_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(self.num_heads*self.head_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k1 = Parameter(torch.Tensor(1, 1, self.num_heads*self.head_dim))
            self.bias_k2 = Parameter(torch.Tensor(1, 1, self.num_heads*self.head_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.num_heads*self.head_dim))
        else:
            self.bias_k1 =self.bias_k2 = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj1.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.k_proj2.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj1.weight)
            nn.init.xavier_uniform_(self.k_proj2.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k1 is not None:
            nn.init.xavier_normal_(self.bias_k1)
            nn.init.xavier_normal_(self.bias_k2)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)


    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ):

        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            return mgk_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj1.bias, self.k_proj2.bias, self.v_proj.bias)),
                self.bias_k1,
                self.bias_k2,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                pi = self.pi,
                scaling = self.scaling,
                dist = self.dist,
                head_dim = self.head_dim,
                training = self.training or self.dropout_module.apply_during_inference,
                key_padding_mask =key_padding_mask,
                need_weights = need_weights,
                attn_mask = attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight1=self.k_proj1.weight,
                k_proj_weight2=self.k_proj2.weight,
                v_proj_weight=self.v_proj.weight,
            )

        # print('query1')
        # import pdb; pdb.set_trace()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k1 = self.k_proj1(query)
            k2 = self.k_proj2(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k1 = k2 = v = None
            else:
                k1 = self.k_proj1(key)
                k2 = self.k_proj2(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k1 = self.k_proj1(key)
            k2 = self.k_proj2(key)
            v = self.v_proj(value)
        # q *= self.scaling

        if (self.bias_k1 is not None) and (self.bias_k2 is not None):
            assert self.bias_v is not None
            k1 = torch.cat([k1, self.bias_k1.repeat(1, bsz, 1)])
            k2 = torch.cat([k2, self.bias_k2.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # print('examine q')
        # import pdb; pdb.set_trace()
        if k1 is not None:
            k1 = (
                k1.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if k2 is not None:
            k2 = (
                k2.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # assert 1==2, 'save states'
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key1" in saved_state:
                _prev_key1 = saved_state["prev_key1"]
                _prev_key2 = saved_state["prev_key2"]
                assert _prev_key1 is not None
                prev_key1 = _prev_key1.view(bsz * self.num_heads, -1, self.head_dim)
                prev_key2 = _prev_key2.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key1
                    k2 = prev_key2
                else:
                    assert k1 is not None
                    assert k2 is not None
                    k1 = torch.cat([prev_key1, k1], dim=1)
                    k2 = torch.cat([prev_key2, k2], dim=1)
                src_len = k1.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k1 is not None and v is not None
            key_padding_mask = MGKAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k1.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key1"] = k1.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key2"] = k2.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k1 is not None
        assert k2 is not None
        assert k1.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k1 = torch.cat([k1, k1.new_zeros((k1.size(0), 1) + k1.size()[2:])], dim=1)
            k2 = torch.cat([k1, k1.new_zeros((k1.size(0), 1) + k1.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        B, Nt, E = q.shape
        Ns = k1.shape[1]

        # self.scaling = self.head_dim ** -0.5
        # attn_weights = torch.bmm(q, k.transpose(1, 2))
        QK1_distance = (-self.scaling/2)*self.dist._sq_dist(q, k1, postprocess = False)
        QK2_distance = (-3*self.scaling/2)*self.dist._sq_dist(q, k2, postprocess = False)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(QK1_distance.size(0), 1, 1)
            QK1_distance += attn_mask
            QK2_distance += attn_mask

        # import pdb; pdb.set_trace()
        # attn_weights = torch.exp(QK1_distance)*torch.clamp(torch.abs(self.pi[0]), min = 1e-6, max = 2.) + torch.exp(QK2_distance)*torch.clamp(torch.abs(self.pi[1]), min = 1e-6, max = 2.)
        attn_weights = torch.exp(QK1_distance).view(B//self.num_heads, -1, Nt, Ns)*torch.clamp(torch.abs(self.pi[0][None, :, None, None]), min = 1e-6, max = 2.) + torch.exp(QK2_distance).view(B//self.num_heads, -1, Nt, Ns)*torch.clamp(torch.abs(self.pi[1][None, :, None, None]), min = 1e-6, max = 2.)
        # print('calculate attn without nn.functional')
        # attn_weights.view(-1, Nt, Ns)

        attn_weights = attn_weights.view(-1, tgt_len, src_len)

        ## this function does nothing
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)


        # import pdb; pdb.set_trace()
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("0."),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("0."))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        ####### I have changed the code here
        if self.onnx_trace:
            attn_weights_float = attn_weights.float()/(attn_weights.float().sum(-1, keepdim = True) + 1e-6)
        else:
            attn_weights_float = attn_weights.to(dtype=torch.float32)/(attn_weights.to(dtype=torch.float32).sum(-1, keepdim = True) + 1e-6)

        # print('haha')  
        # import pdb; pdb.set_trace()

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.num_heads*self.head)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads*self.head_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        
        return attn, attn_weights


    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj1.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "k_proj2.weight"] = state_dict[k][2 * dim: 3*dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][3 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim * 2 : 3 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][3 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


def mgk_attention_forward(query,key,value, embed_dim_to_check, 
    num_heads, in_proj_weight, in_proj_bias, bias_k1,bias_k2, bias_v,
    add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,pi, scaling, dist,head_dim,
    training = True, key_padding_mask = None, need_weights = True,
    attn_mask = None, use_separate_proj_weight = False,
    q_proj_weight = None,
    k_proj_weight1 = None,
    k_proj_weight2 = None,
    v_proj_weight = None,
    static_k1 = None, static_k2 = None, static_v= None):

    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k1,bias_k2 , bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        assert 1==2
        return handle_torch_function(
            mgk_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k1,
            bias_k2,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight1=k_proj_weight1,
            k_proj_weight2=k_proj_weight2,
            v_proj_weight=v_proj_weight,
            static_k1=static_k1,
            static_k2=static_k2,
            static_v=static_v,
            pi = pi,
            scaling = scaling
        )
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    # if isinstance(embed_dim, torch.Tensor):
    #     # embed_dim can be a tensor when JIT tracing
    #     head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    # else:
    #     head_dim = embed_dim // num_heads
    # assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert 1==2, 'check on this'
        q, k1, k2, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight1 is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert k_proj_weight2 is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k1 = b_k2 = b_v = None
        else:
            b_q, b_k1, b_k2, b_v = in_proj_bias.chunk(4)
        q, k1 ,k2, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight1, k_proj_weight2, v_proj_weight, b_q, b_k1, b_k2, b_v)

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in MGKAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in MGKAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k1 is not None and bias_v is not None:
        assert static_k1 is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k2 = torch.cat([k1, bias_k1.repeat(1, bsz, 1)])
        k2 = torch.cat([k2, bias_k2.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k1 is None
        assert bias_k2 is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k1 is None:
        k1 = k1.contiguous().view(k1.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        k2 = k2.contiguous().view(k2.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k1.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k1.size(0)}"
        assert static_k1.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k1.size(2)}"

        assert static_k2.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k2.size(0)}"
        assert static_k2.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k2.size(2)}"
        k1 = static_k1
        k2 = static_k2
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k1 = torch.cat([k1, torch.zeros(zero_attn_shape, dtype=k1.dtype, device=k1.device)], dim=1)
        k2 = torch.cat([k2, torch.zeros(zero_attn_shape, dtype=k2.dtype, device=k2.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k1.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = rbf2keys_attention(q, k1, k2, v,pi, scaling,dist, num_heads, attn_mask, dropout_p)
    # print(attn_output_weights)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, num_heads*head_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    
    if need_weights:
        # average attention weights over heads
        # import pdb; pdb.set_trace()
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

def _in_projection_packed(
    q, k, v, w, b = None):

    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(4, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kkv = w.split([E, E * 3])
            if b is None:
                b_q = b_kkv = None
            else:
                b_q, b_kkv = b.split([E, E * 3])
            return (linear(q, w_q, b_q),) + linear(k, w_kkv, b_kkv).chunk(3, dim=-1)
    else:
        w_q, w_k1, w_k2, w_v = w.chunk(4)
        if b is None:
            b_q = b_k1, b_k2 = b_v = None
        else:
            b_q, b_k1, b_k2, b_v = b.chunk(4)
        return linear(q, w_q, b_q), linear(k, w_k1, b_k1), linear(k, w_k2, b_k2), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k1: Tensor,
    w_k2: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k1: Optional[Tensor] = None,
    b_k2: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
):
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    # assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    # assert w_k1.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k1.shape}"
    # assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    # assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    # assert b_k1 is None or b_k1.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k1.shape}"
    # assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k1, b_k1),linear(k, w_k2, b_k2), linear(v, w_v, b_v)



