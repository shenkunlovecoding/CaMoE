"""
ROSA experimental module for CaMoE.

This file provides:
1) A pure-python 1-bit ROSA sequence transform.
2) A torch module that maps hidden states -> binary streams -> ROSA -> embedding.
"""

from typing import List

import torch
import torch.nn as nn


def rosa_1bit_sequence(x: List[int]) -> List[int]:
    r"""rosa_1bit_sequence(x) -> List[int]

    1-bit ROSA transform (token set {0, 1}) based on the online suffix-automaton style
    draft algorithm shared in RWKV-8 ROSA notes.

    For each position i, the output y[i] tries to predict the next token after the
    best historical match; returns -1 when no valid match is found.
    """
    n = len(x)
    y = [-1] * n

    # Allocate enough states for online extension.
    s = 2 * n + 1
    b = [None] * s  # transitions
    c = [-1] * s    # suffix links
    d = [0] * s     # state lengths
    e = [-1] * s    # latest end position
    b[0] = {}

    g = 0  # active state
    z = 1  # next state id

    for i, t in enumerate(x):
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g

        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]

        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p].get(t) == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = u
                c[r] = u

        # Query best available "next token" candidate.
        v = g
        a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                nxt = e[v] + 1
                if nxt < n:
                    a = x[nxt]
                break
            v = c[v]
        y[i] = a

        # Update end positions along suffix links.
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]

        g = r

    return y


class ROSA1bitLayer(nn.Module):
    r"""ROSA1bitLayer(n_embd, num_streams=32, rosa_emb_dim=64) -> None

    Experimental ROSA layer:
    - projects hidden states to multiple binary streams
    - runs per-stream 1-bit ROSA in python
    - embeds ROSA outputs and projects back to model dim
    """

    def __init__(
        self,
        n_embd: int,
        num_streams: int = 32,
        rosa_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_streams = num_streams

        self.to_bits = nn.Linear(n_embd, num_streams, bias=False)
        # token ids: 0,1 and 2 for "no match" (-1)
        self.rosa_emb = nn.Embedding(3, rosa_emb_dim)
        self.out = nn.Linear(rosa_emb_dim, n_embd, bias=False)
        self.norm = nn.LayerNorm(n_embd)

        with torch.no_grad():
            nn.init.normal_(self.rosa_emb.weight, std=0.02)
            nn.init.orthogonal_(self.out.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Args:
          x (Tensor): [B, T, C]

        Returns:
          Tensor: [B, T, C], ROSA feature branch output.
        """
        B, T, _ = x.shape
        bit_logits = self.to_bits(x)  # [B, T, S]
        bits = (bit_logits > 0).to(torch.long)

        # Run ROSA on CPU-side python lists (experimental reference implementation).
        rosa_tokens = torch.empty_like(bits)
        for b in range(B):
            for s in range(self.num_streams):
                seq = bits[b, :, s].tolist()
                y = rosa_1bit_sequence(seq)
                mapped = [2 if t < 0 else t for t in y]
                rosa_tokens[b, :, s] = torch.tensor(mapped, dtype=torch.long, device=x.device)

        # Aggregate multi-stream ROSA outputs into model space.
        emb = self.rosa_emb(rosa_tokens)     # [B, T, S, D]
        pooled = emb.mean(dim=2)             # [B, T, D]
        out = self.out(pooled)               # [B, T, C]
        return self.norm(out)

