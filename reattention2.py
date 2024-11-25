from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# # isort: off
# # We need to import the CUDA kernels after importing torch
# # Use relative import to support build-from-source installation in vLLM
# from . import vllm_flash_attn_c # noqa: F401

# # isort: on

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x



def _flash_attn_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    block_table,
    *,
    out=None
):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = torch.ops.vllm_flash_attn_c.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state



def main():
    # 定义输入张量大小和参数
    batch_size = 1
    seq_len_q = 16
    seq_len_k = 16
    num_heads = 32
    head_dim = 128

    # 生成随机输入张量
    q = torch.rand((batch_size * num_heads, seq_len_q, head_dim), device='cuda')
    k = torch.rand((batch_size * num_heads, seq_len_k, head_dim), device='cuda')
    v = torch.rand((batch_size * num_heads, seq_len_k, head_dim), device='cuda')

    # 累积序列长度，用于不定长序列的处理
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len_q, seq_len_q, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seq_len_k, seq_len_k, dtype=torch.int32, device='cuda')

    # 其他参数
    max_seqlen_q = seq_len_q
    max_seqlen_k = seq_len_k
    dropout_p = 0.1
    softmax_scale = 1.0 / (head_dim ** 0.5)
    causal = False
    window_size = (1, 1)  # 假设窗口大小为 (1, 1)
    softcap = 0.0
    alibi_slopes = torch.zeros(num_heads, device='cuda')  # 这里简单设置为零，可以根据需要调整
    return_softmax = False
    block_table = None

    # 调用 _flash_attn_varlen_forward 函数
    out, q_out, k_out, v_out, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax, block_table
    )

    # 打印输出信息
    print("Output Shape:", out.shape)
    print("Q Output Shape:", q_out.shape)
    print("K Output Shape:", k_out.shape)
    print("V Output Shape:", v_out.shape)

if __name__ == "__main__":
    main()
