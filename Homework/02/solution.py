import torch
import torch.nn.functional as F


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM = queries.shape

    scaling_factor = HIDDEN_DIM ** 0.5
    keys_transposed = keys.transpose(1, 2)
    attention_scores = torch.bmm(queries, keys_transposed) / scaling_factor
    attention_probs = F.softmax(attention_scores, dim=-1)

    return torch.bmm(attention_probs, values)


def compute_multihead_attention(queries, keys, values, projection_matrix):
    """
    Compute multi-head attention using double precision.

    queries: (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys: (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values: (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix: (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """

    attention_output = F.scaled_dot_product_attention(queries, keys, values)
    BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD = queries.shape
    concat_attention_output = attention_output.permute(0, 2, 1, 3).reshape(BATCH_SIZE, SEQ_LENGTH, -1)
    final_output = torch.matmul(concat_attention_output, projection_matrix.T)

    return final_output


def compute_rotary_embeddings(x) -> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD = x.shape

    positions = torch.arange(SEQ_LENGTH, dtype=x.dtype, device=x.device).unsqueeze(1)
    dim_index = torch.arange(DIM_PER_HEAD // 2, dtype=x.dtype, device=x.device).unsqueeze(0)

    inv_freq = 1.0 / (10000 ** (2 * dim_index / DIM_PER_HEAD))
    theta = positions * inv_freq

    sin_theta = torch.sin(theta).unsqueeze(0).unsqueeze(2)
    cos_theta = torch.cos(theta).unsqueeze(0).unsqueeze(2)

    x_reshaped = x.view(BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD // 2, 2)
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]

    x_rotated_real = x_real * cos_theta - x_imag * sin_theta
    x_rotated_imag = x_real * sin_theta + x_imag * cos_theta
    x_rotated = torch.stack((x_rotated_real, x_rotated_imag), dim=-1).reshape(BATCH_SIZE, SEQ_LENGTH, N_HEADS,
                                                                              DIM_PER_HEAD)
    return x_rotated
