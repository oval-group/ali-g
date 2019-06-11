import torch


@torch.autograd.no_grad()
def l2_projection(parameters, max_norm):
    if max_norm is None:
        return
    total_norm = torch.sqrt(sum(p.norm() ** 2 for p in parameters))
    if total_norm > max_norm:
        ratio = max_norm / total_norm
        for p in parameters:
            p *= ratio
