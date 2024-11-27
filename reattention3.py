from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.vllm_flash_attn
import time

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




class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        deterministic,
        return_softmax,
        block_table,
        out=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
            out=out,
        )
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)
    



def main():
    # 定义输入张量大小和参数
    batch_size = 1
    seq_len_q = 16
    seq_len_k = 16
    num_heads = 32
    head_dim = 128

    # 生成随机输入张量
    q = torch.rand((seq_len_q, batch_size * num_heads, head_dim), device='cuda',dtype=torch.float32)
    k = torch.rand((seq_len_k, batch_size * num_heads, head_dim), device='cuda',dtype=torch.float32)
    v = torch.rand((seq_len_k, batch_size * num_heads, head_dim), device='cuda',dtype=torch.float32)
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    # 累积序列长度，用于不定长序列的处理
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len_q, seq_len_q, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seq_len_k, seq_len_k, dtype=torch.int32, device='cuda')

    # 其他参数
    max_seqlen_q = seq_len_q
    max_seqlen_k = seq_len_k
    dropout_p = 0
    softmax_scale = 1.0 / (head_dim ** 0.5)
    causal = False
    window_size = (1, 1)  # 假设窗口大小为 (1, 1)
    softcap = 0.0
    # alibi_slopes = torch.zeros(num_heads, device='cuda')  # 这里简单设置为零，可以根据需要调整
    alibi_slopes = None
    return_softmax = False
    block_table = None
    start_time = time.time()
    # 调用 _flash_attn_varlen_forward 函数
    out, q_out, k_out, v_out, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax, block_table
    )
    end_time = time.time()
    # 打印输出信息
    print("Output Shape:", out.shape)
    print("Q Output Shape:", q_out.shape)
    print("K Output Shape:", k_out.shape)
    print("V Output Shape:", v_out.shape)
    elapsed_time = (end_time - start_time)*1000
    flops_qk = 2 * seq_len_q * batch_size * num_heads * head_dim * seq_len_k
    flops_output = 2 * seq_len_q * batch_size * num_heads * head_dim * seq_len_k
    total_flops = flops_qk + flops_output
    gflops = total_flops / 1e9
    print(f"Total FLOPs: {total_flops:,} FLOPs")
    print(f"Total Gflops: {gflops:.4f} Gflops")
    print(f"Elapsed Time: {elapsed_time:.4f} ms")
   
if __name__ == "__main__":
    main()
