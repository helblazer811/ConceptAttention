import torch
from torch import nn, Tensor
from einops import rearrange

from concept_attention.flux.src.flux.modules.layers import Modulation, QKNorm
from concept_attention.flux.src.flux.math import attention

NUM_IMAGE_PATCHES = 4096

class ModifiedSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        cache_vectors=False
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)


        # Store vectors
        self.concept_output_vectors = []
        self.image_output_vectors = []
        self.cache_vectors = cache_vectors

    def clear_cached_vectors(self):
        self.concept_output_vectors = []
        self.image_output_vectors = []

    def forward(self, x: Tensor, concepts: Tensor, vec: Tensor, pe: Tensor, concept_pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)

        # Perform img-text self attention
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        ##########################################################
        num_concepts = concepts.shape[1]
        # Unpack the original image attn vectors
        # img_attn = attn[:, -NUM_IMAGE_PATCHES:]
        # Now perform image/concept attention
        img = x[:, -NUM_IMAGE_PATCHES:]
        img_concept_x = torch.cat([concepts, img], dim=1)
        img_concepts_mod = (1 + mod.scale) * self.pre_norm(img_concept_x) + mod.shift
        qkv, mlp = torch.split(self.linear1(img_concepts_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn_img_concept = attention(q, k, v, pe=concept_pe)
        img_output_vectors = attn_img_concept[:, -NUM_IMAGE_PATCHES:]
        concept_output_vectors = attn_img_concept[:, :-NUM_IMAGE_PATCHES]
        if self.cache_vectors:
            self.concept_output_vectors.append(concept_output_vectors.detach().cpu())
            self.image_output_vectors.append(img_output_vectors.detach().cpu())
        # Now do the second linear layer
        output_img_concept = self.linear2(torch.cat((attn_img_concept, self.mlp_act(mlp)), 2))
        concept_output = output_img_concept[:, :-NUM_IMAGE_PATCHES]
        ##########################################################
        
        return x + mod.gate * output, concepts + mod.gate * concept_output
