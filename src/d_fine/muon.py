"""Single-device Muon optimizer with an AdamW auxiliary path.

Muon (Jordan et al., 2024): Newton-Schulz-orthogonalized momentum for 2D weight
matrices. Only the encoder/decoder attention/MLP linears are routed to Muon; the
backbone, norms, biases, embeddings and det/mask heads stay on AdamW.

One optimizer object drives both paths (param groups flagged ``use_muon``), so the
train loop's step / zero_grad / AMP / scheduler code is unchanged. Param groups may
carry extra keys (e.g. OneCycleLR's ``initial_lr``/``max_lr``) — we read only what we
need. Adapted from Keller Jordan's modded-nanogpt single-device reference.
"""
import torch


def _zeropower_via_newtonschulz5(G, steps=5):
    # Quintic Newton-Schulz iteration -> approximate orthogonalization of G (bf16).
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def _muon_update(grad, momentum, beta=0.95, ns_steps=5):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta)  # nesterov
    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(0) / grad.size(1)) ** 0.5  # match RMS to fan shape
    return update


def _adamw_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    bias1 = 1 - betas[0] ** step
    bias2 = 1 - betas[1] ** step
    return (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Per-group: ``use_muon=True`` -> Muon update, else AdamW. Single device."""

    def __init__(self, param_groups):
        for g in param_groups:
            if g.get("use_muon"):
                g.setdefault("momentum", 0.95)
            else:
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-8)
            g.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g.get("use_muon"):
                    if not st:
                        st["momentum_buffer"] = torch.zeros_like(p)
                    upd = _muon_update(p.grad, st["momentum_buffer"], beta=g["momentum"])
                    p.mul_(1 - lr * wd)
                    p.add_(upd.reshape(p.shape), alpha=-lr)
                else:
                    if not st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    upd = _adamw_update(
                        p.grad, st["exp_avg"], st["exp_avg_sq"], st["step"], g["betas"], g["eps"]
                    )
                    p.mul_(1 - lr * wd)
                    p.add_(upd, alpha=-lr)
        return loss
