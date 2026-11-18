import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP


class MultimodalMoELayer(nn.Module):
    """Sparse Mixture-of-Experts Layer operating on the fused multimodal sequence.

    Positioned after Multimodal Fusion (concatenation of visual and textual tokens)
    and before the Language Model Decoder.

    Pipeline:
        1. LayerNorm on input tokens.
        2. Router calculates gating logits for each token across E experts.
        3. Top-K gating selects K experts per token with renormalized weights.
        4. Tokens are routed to selected expert MLPs.
        5. Outputs are aggregated via weighted sum and combined with residual connection.
        6. Computes auxiliary load balancing loss to prevent router collapse.
    """

    def __init__(
        self,
        d_model: int = 896,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize MultimodalMoELayer.

        Args:
            d_model: Token embedding dimensionality (matching d_llm).
            num_experts: Total number of parallel Expert MLPs (E).
            top_k: Number of activated experts per token (K <= E).
            dropout: Dropout probability inside each expert MLP.
        """
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k must be between 1 and num_experts"

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.layer_norm = nn.LayerNorm(d_model)
        # Gating router: maps token representation to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert pool: E independent feed-forward networks
        self.experts = nn.ModuleList([
            MLP(
                in_features=d_model,
                out_features=d_model,
                dropout=dropout,
                expansion_factor=4,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """Forward pass with Top-K sparse routing and auxiliary loss computation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, seq_len, d_model).
                - Auxiliary load balancing loss scalar.
        """
        orig_shape = x.shape
        B, T, C = orig_shape

        # Flatten batch and sequence into a continuous token dimension: (B * T, C)
        # Ensure input dtype matches layer_norm parameters
        flat_x = x.view(-1, C).to(dtype=self.layer_norm.weight.dtype)
        num_tokens = flat_x.shape[0]

        normed_x = self.layer_norm(flat_x)

        # Step 1: Compute router logits and Softmax probabilities across all experts
        router_logits = self.router(normed_x)  # Shape: (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Step 2: Select Top-K experts per token
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (num_tokens, top_k)

        # Renormalize top-k probabilities so they sum to 1.0 across selected experts
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)

        # Step 3: Compute Auxiliary Load Balancing Loss (Switch Transformer formulation)
        # Prevents router collapse by encouraging uniform distribution across experts.
        # L_aux = num_experts * sum(fraction_tokens_i * mean_prob_i)
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()  # (num_tokens, top_k, E)
        token_dispatch = expert_mask.sum(dim=1)  # (num_tokens, E)
        fraction_per_expert = token_dispatch.mean(dim=0)  # Shape: (E,)
        mean_prob_per_expert = router_probs.mean(dim=0)  # Shape: (E,)
        aux_loss = self.num_experts * torch.sum(fraction_per_expert * mean_prob_per_expert)

        # Step 4: Dispatch tokens to selected experts and compute weighted output
        expert_out = torch.zeros_like(flat_x)

        for expert_idx, expert in enumerate(self.experts):
            # Locate tokens where this expert was selected among top-K
            token_idx, k_rank = torch.where(topk_indices == expert_idx)
            if token_idx.numel() == 0:
                continue

            # Route tokens through the expert
            selected_tokens = flat_x[token_idx]
            out = expert(selected_tokens)

            # Weight expert output by gating coefficient
            weights = topk_weights[token_idx, k_rank, None]  # Shape: (matched_tokens, 1)
            expert_out.index_add_(0, token_idx, out * weights)

        # Step 5: Add residual connection and restore original tensor shape (B, T, C)
        final_out = flat_x + expert_out
        return final_out.view(orig_shape).to(dtype=x.dtype), aux_loss
