import jax
import jax.numpy as jnp
import flax.linen as nn

import math

class SigmoidAttention(nn.Module):
    dim:int
    head_dim:int
    num_hds:int = 10
    qkv_bias:bool = True
    use_qk_norm: bool = True
    use_layerscale: bool = True
    expected_k: float = 1.0
    layerscale_init: float = 1e-3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        B, N, C = x.shape
        scale = self.head_dim ** -.5

        expected_k = self.expected_k

        qkv = nn.Dense(self.dim * 3, use_bias = self.qkv_bias, dtype = self.dtype)(x)
        q = nn.Dense(self.num_hds * self.head_dim, dtype = self.dtype, name="q_proj")(x)
        k = nn.Dense(self.num_hds * self.head_dim, dtype = self.dtype, name="k_proj")(x)
        v = nn.Dense(self.num_hds * self.head_dim, dtype = self.dtype, name="v_proj")(x)
        #q, k, v = jnp.split(qkv, 3, axis = -1)

        q = q.reshape(B, N, self.num_hds, self.head_dim)
        k = k.reshape(B, N, self.num_hds, self.head_dim)
        v = v.reshape(B, N, self.num_hds, self.head_dim)

        if self.use_qk_norm:
          q = nn.LayerNorm(name='q_norm')(q)
          k = nn.LayerNorm(name='k_norm')(k)

        # (Batch, NumHeads, SeqLen, HeadDim) -> Transpose for dot product
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)

        # Logits: (B, H, N, N)
        attn_logits = jnp.einsum('bhid, bhjd->bhij', q, k)

        exp = expected_k / N
        print(f"exp:{exp}")
        bias = math.log(exp/(1 - exp))
        print(bias)
        init_bias = self.param('init_bias',
                          nn.initializers.constant(bias), (1, 1, 1, 1))
        """init_bias = self.param('init_bias',
                          nn.initializers.constant(self.init_bias), (1, self.num_hds, 1, 1))"""

        # print(f"attn_logits: {attn_logits.shape}, bias: {init_bias.shape}")
        attn_weights = jax.nn.sigmoid(attn_logits + init_bias)
        # print(f"attn_weights: {attn_weights.shape}")

        if mask is not None:
            attn_weights = jax.nn.sigmoid(attn_weights + init_bias)
            # print(f"attn_weights: {attn_weights.shape}")

        out = jnp.einsum('bhij, bhjd -> bhid', attn_weights, v)
        out = jnp.swapaxes(out, 1, 2).reshape(B, N, -1)

        out = nn.Dense(self.dim, dtype=self.dtype)(out)
        # print(f"self.dim: {self.dim}")
        # print(f"out:{out.shape}")

        if self.use_layerscale:
            gamma = self.param('gamma',
                                nn.initializers.constant(self.layerscale_init),
                                (self.dim,))

            out = out * gamma

        return out

def make_causal_mask(B, H, N):
    base = jnp.tril(jnp.ones((N, N), dtype=bool))
    mask = base[None, None, :, :]
    mask = jnp.broadcast_to(mask, (B, H, N, N))
    return mask

if __name__ == "__main__":

  rng = jax.random.PRNGKey(0)
  x = jax.random.normal(rng, (2, 1024, 512))

  init_fn = jax.jit(SigmoidAttention(dim=512, num_hds=8, head_dim=64).init)
  apply_fn = jax.jit(SigmoidAttention(dim=512, num_hds=8, head_dim=64).apply, static_argnames=['mutable'])

  variables = init_fn(rng, x)

  output = apply_fn({'params': variables['params']}, x)

  print(f"Input Shape: {x.shape}")
  print(f"Output Shape: {output.shape}")

  val = variables['params']['init_bias']
  exp_val = -math.log(1024)
  print(f"Initialized Bias: {val}, Expected Bias: {exp_val}")

  N = 1024
  make_causal_mask(1, 1, N)
  # y, mutations = apply_fn({'params': variables['params']}, x, mutable=('batch_stats', 'intermediates'))