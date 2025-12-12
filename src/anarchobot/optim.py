import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to approximate the orthogonal factor of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, ns_steps: int = 5, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-Schulz.
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(grad, state["momentum_buffer"], beta=beta)
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class MuonWithAdam(torch.optim.Optimizer):
    """
    Single-device Muon combined with AdamW for embeddings/bias terms.
    Pass param groups with the key `use_muon`: True/False.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = group["lr"]
                wd = group["weight_decay"]
                beta = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(grad, state["momentum_buffer"], beta=beta)
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
            else:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                    bias_c1 = 1 - beta1 ** state["step"]
                    bias_c2 = 1 - beta2 ** state["step"]
                    denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
                    step = (exp_avg / bias_c1) / denom
                    p.mul_(1 - lr * wd)
                    p.add_(step, alpha=-lr)
        return loss


def build_muon_adam_optimizer(model, lr_muon: float, lr_adam: float, weight_decay: float, momentum: float = 0.95):
    """
    Partition parameters into Muon-suitable (matrices) and AdamW (embeddings/bias) groups.
    """
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    param_groups = [
        dict(params=muon_params, lr=lr_muon, weight_decay=weight_decay, momentum=momentum, use_muon=True),
        dict(params=adam_params, lr=lr_adam, weight_decay=weight_decay, use_muon=False),
    ]
    return MuonWithAdam(param_groups)
