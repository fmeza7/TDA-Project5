from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    # torch.cdist handles batching internally
    return torch.cdist(x, x, p=2)


def mst_edge_indices(dist_matrix: torch.Tensor) -> torch.LongTensor:
    num_nodes = dist_matrix.shape[0]
    if num_nodes <= 1:
        return torch.empty((0, 2), dtype=torch.long, device=dist_matrix.device)
    # Use Prim's algorithm on CPU numpy for stability
    dist_np = dist_matrix.detach().cpu().numpy()
    selected = np.zeros(num_nodes, dtype=bool)
    selected[0] = True
    edges = []
    for _ in range(num_nodes - 1):
        best = (None, None, np.inf)
        for i in range(num_nodes):
            if not selected[i]:
                continue
            for j in range(num_nodes):
                if selected[j]:
                    continue
                w = dist_np[i, j]
                if w < best[2]:
                    best = (i, j, w)
        i, j, _ = best
        edges.append((i, j))
        selected[j] = True
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=dist_matrix.device)
    return edge_tensor


def gather_edge_distances(dist_matrix: torch.Tensor, edges: torch.LongTensor) -> torch.Tensor:
    if edges.numel() == 0:
        return torch.zeros(0, device=dist_matrix.device, dtype=dist_matrix.dtype)
    return dist_matrix[edges[:, 0], edges[:, 1]]


def topo_loss_0d(x_input: torch.Tensor, z_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ax = pairwise_distances(x_input)
    az = pairwise_distances(z_latent)
    edges_x = mst_edge_indices(ax)
    edges_z = mst_edge_indices(az)
    dist_x = gather_edge_distances(ax, edges_x)
    dist_z = gather_edge_distances(az, edges_x)
    loss_xz = 0.5 * torch.mean((dist_x - dist_z) ** 2) if dist_x.numel() else torch.tensor(0.0, device=x_input.device)
    dist_z2 = gather_edge_distances(az, edges_z)
    dist_x2 = gather_edge_distances(ax, edges_z)
    loss_zx = 0.5 * torch.mean((dist_z2 - dist_x2) ** 2) if dist_z2.numel() else torch.tensor(0.0, device=x_input.device)
    return loss_xz + loss_zx, loss_xz, loss_zx
