import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    dim_key = K.shape[-1]

    attention_scores = Q @ K.T / np.sqrt(dim_key)

    attn_weights = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

    return attn_weights @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # 获取输入的形状
    seq_len, d_model = Q.shape
    # 计算每个头的维度
    d_k = d_model // n_heads

    # 初始化一个列表来存储每个头的输出
    head_outputs = []
    # 遍历每个头，切出第 i 个头对应的列范围，计算自注意力，将输出添加到列表中
    for i in range(n_heads):
        # 切出第 i 个头对应的列范围
        start, end = i * d_k, (i + 1) * d_k
        # 计算第 i 个头的自注意力输出
        head_output = self_attention(Q[:, start:end], K[:, start:end], V[:, start:end])
        # 将输出添加到列表中
        head_outputs.append(head_output)

    # 把所有头的输出沿最后一个维度拼接，恢复 (seq_len, d_model)
    return np.concatenate(head_outputs, axis=-1)