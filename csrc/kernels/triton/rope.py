"""RoPE, written in Triton instead of raw CUDA. UNVERIFIED IN THIS SESSION
-- no GPU (and so no Triton runtime) was available while writing this; see
docs/learning/phase-07.md. This is the Triton half of ADR-007's "one
kernel, both ways" comparison against csrc/kernels/cuda/rope.cu -- same
math, written in a much shorter, higher-level form, since Triton handles a
lot of the low-level indexing bookkeeping that the raw CUDA version has to
spell out by hand.
"""
import triton
import triton.language as tl


@triton.jit
def rope_kernel(x_ptr, positions_ptr, n_heads, head_dim, theta,
                 BLOCK_SIZE: tl.constexpr):
    # Triton programs are organized similarly to CUDA blocks: this program
    # instance handles one (token, head) pair, and BLOCK_SIZE numbers
    # within it -- Triton just expresses "a whole vector of numbers at
    # once" more directly than CUDA's one-thread-per-number style.
    token = tl.program_id(0)
    head = tl.program_id(1)
    half = head_dim // 2

    pair_index = tl.arange(0, BLOCK_SIZE)
    mask = pair_index < half

    position = tl.load(positions_ptr + token).to(tl.float32)
    freq = tl.exp(-2.0 * pair_index.to(tl.float32) / head_dim * tl.log(theta))
    angle = position * freq
    cos_a = tl.cos(angle)
    sin_a = tl.sin(angle)

    base = token * n_heads * head_dim + head * head_dim
    x0 = tl.load(x_ptr + base + pair_index, mask=mask)
    x1 = tl.load(x_ptr + base + pair_index + half, mask=mask)

    tl.store(x_ptr + base + pair_index, x0 * cos_a - x1 * sin_a, mask=mask)
    tl.store(x_ptr + base + pair_index + half, x0 * sin_a + x1 * cos_a,
             mask=mask)


def apply_rope_triton(x, positions, n_tokens, n_heads, head_dim, theta):
    """x is a flat GPU buffer of n_tokens * n_heads * head_dim floats,
    modified in place -- same contract as the raw CUDA and CPU versions.
    """
    half = head_dim // 2
    block_size = triton.next_power_of_2(half)
    grid = (n_tokens, n_heads)
    rope_kernel[grid](x, positions, n_heads, head_dim, theta,
                       BLOCK_SIZE=block_size)
