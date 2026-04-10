import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    # 获取 key 的最后一维维度 dim_key
    # key的shape：
    #   常见：三维：(batch_size, seq_len, dim_key)
    #   多头：四维：(batch_size, n_heads, seq_len, dim_key)
    dim_key = key.shape[-1]

    # 计算 query 和 key^T 的点积
    attention_scores = query @ key.transpose(-2, -1) # transpose:转置：把张量的倒数第 2 个维度和最后一个维度交换位置

    # 缩放 attention_scores，除以 dim_key 的平方根
    attention_scores = attention_scores / torch.sqrt(torch.tensor(dim_key, dtype=torch.float32))

    # 对 attention_scores 应用 softmax 函数，得到注意力权重
    attention_weights = F.softmax(attention_scores, dim=-1) # softmax:会对 attention_scores 的每一行（最后一维）做归一化，让每一组得分变成概率

    