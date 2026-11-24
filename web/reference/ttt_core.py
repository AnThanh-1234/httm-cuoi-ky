# ==== TTT layer (Sun et al., "Learning to (Learn at Test Time)") dùng làm memory phụ ====
# Cài đặt riêng theo mô tả trong paper (không phải code chính thức):
#   loss      l(W; x) = 1/2 || f(theta_K x; W) - theta_V x ||^2           (Eq. 4, hệ số 1/2 gộp vào lr)
#   f(x; W)   = x + LN(f_res(x; W)),  f_res = Linear (TTT-Linear) hoac MLP 2 lop GELU, hidden = 4 * dim (TTT-MLP)
#   mini-batch TTT (b = block): gradient cua moi token trong block tinh tai W dau block (t' = t - mod(t, b));
#       W_t = W_t' - sum_{s=t'+1..t} eta_s * grad l(W_t'; x_s)             (mini-batch GD)
#   output    z_t = f(theta_Q x_t; W_t)                                    (Eq. 5)
#   eta(x)    = eta_base * sigmoid(theta_lr . x)  (eta_base = 1 Linear, 0.1 MLP);  W_0 hoc duoc (outer loop)
# Tinh toan bang "dual form": khong tao W_t rieng cho tung token, ket qua tuong duong ve mat toan hoc.
# Ghi chu: chia eta cho head_dim la lua chon de on dinh (theo thong le cua cac ban cai dat tham khao), khong co trong paper.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def ln_fwd(x, gamma, beta, eps=1e-6):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta


def ln_l2_bwd(x, target, gamma, beta, eps=1e-6):
    # gradient theo x cua 1/2 || LN(x) - target ||^2 (dang dong kin)
    D = x.shape[-1]
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    g = (y - target) * gamma
    return (D * g - g.sum(-1, keepdim=True) - x_hat * (g * x_hat).sum(-1, keepdim=True)) / (D * std)


def gelu_grad(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + x * torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _lower_eta(eta, k):
    # eta: (B, nh, k) -> E[t, s] = eta_s voi s <= t
    return torch.tril(eta.unsqueeze(-2).expand(*eta.shape[:-1], k, k))


def ttt_linear_block(st, q, k, v, eta, ln_w, ln_b):
    # q, k, v: (B, nh, kk, f); eta: (B, nh, kk); st: W1 (B,nh,f,f), b1 (B,nh,1,f)
    W, b = st["W1"], st["b1"]
    kk = k.size(-2)
    Z1 = k @ W + b
    G = ln_l2_bwd(Z1, v - k, ln_w, ln_b)
    loss_tok = ((k + ln_fwd(Z1, ln_w, ln_b) - v) ** 2).mean(-1)          # (B, nh, kk) "surprise"
    E = _lower_eta(eta, kk)
    A = torch.tril(q @ k.transpose(-1, -2))
    Zb = q @ W - (E * A) @ G + (b - E @ G)
    out = q + ln_fwd(Zb, ln_w, ln_b)
    W_new = W - (eta.unsqueeze(-1) * k).transpose(-1, -2) @ G
    b_new = b - (eta.unsqueeze(-1) * G).sum(-2, keepdim=True)
    return out, {"W1": W_new, "b1": b_new}, loss_tok


def ttt_mlp_block(st, q, k, v, eta, ln_w, ln_b):
    W1, b1, W2, b2 = st["W1"], st["b1"], st["W2"], st["b2"]
    kk = k.size(-2)
    Z1 = k @ W1 + b1
    X2 = F.gelu(Z1)
    Z2 = X2 @ W2 + b2
    G2 = ln_l2_bwd(Z2, v - k, ln_w, ln_b)
    G1 = (G2 @ W2.transpose(-1, -2)) * gelu_grad(Z1)
    loss_tok = ((k + ln_fwd(Z2, ln_w, ln_b) - v) ** 2).mean(-1)
    E = _lower_eta(eta, kk)
    A1 = torch.tril(q @ k.transpose(-1, -2))
    Z1b = q @ W1 - (E * A1) @ G1 + (b1 - E @ G1)
    X2b = F.gelu(Z1b)
    A2 = torch.tril(X2b @ X2.transpose(-1, -2))
    Z2b = X2b @ W2 - (E * A2) @ G2 + (b2 - E @ G2)
    out = q + ln_fwd(Z2b, ln_w, ln_b)
    e = eta.unsqueeze(-1)
    new = {
        "W1": W1 - (e * k).transpose(-1, -2) @ G1, "b1": b1 - (e * G1).sum(-2, keepdim=True),
        "W2": W2 - (e * X2).transpose(-1, -2) @ G2, "b2": b2 - (e * G2).sum(-2, keepdim=True),
    }
    return out, new, loss_tok


def flat_state(state):
    return torch.cat([state[n].flatten(1) for n in sorted(state)], dim=-1)


class TTTMem(nn.Module):
    """Giao dien: forward(x:(B,T,dim), state, want_trace) -> (read:(B,T,dim), new_state, info)."""

    def __init__(self, dim, heads=4, head_dim=64, block=16, kind="linear", eta_base=None):
        super().__init__()
        assert kind in ("linear", "mlp")
        self.nh, self.f, self.b, self.kind = heads, head_dim, block, kind
        self.eta_base = eta_base if eta_base is not None else (1.0 if kind == "linear" else 0.1)
        inner = heads * head_dim
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_lr = nn.Linear(dim, heads)
        nn.init.normal_(self.to_lr.weight, std=0.02)
        nn.init.zeros_(self.to_lr.bias)
        self.merge = nn.Linear(inner, dim, bias=False)
        self.ln_w = nn.Parameter(torch.ones(heads, 1, head_dim))
        self.ln_b = nn.Parameter(torch.zeros(heads, 1, head_dim))
        f = head_dim
        if kind == "linear":
            self.W1 = nn.Parameter(torch.randn(heads, f, f) * 0.02)
            self.b1 = nn.Parameter(torch.zeros(heads, 1, f))
            self.state_names = ["W1", "b1"]
        else:
            h = 4 * f
            self.W1 = nn.Parameter(torch.randn(heads, f, h) * 0.02)
            self.b1 = nn.Parameter(torch.zeros(heads, 1, h))
            self.W2 = nn.Parameter(torch.randn(heads, h, f) * 0.02)
            self.b2 = nn.Parameter(torch.zeros(heads, 1, f))
            self.state_names = ["W1", "b1", "W2", "b2"]

    def init_state(self, B):
        return {n: getattr(self, n).unsqueeze(0).expand(B, *getattr(self, n).shape).clone() for n in self.state_names}

    def forward(self, x, state=None, want_trace=False):
        B, T, _ = x.shape
        h = self.norm(x)
        sp = lambda t: t.view(B, T, self.nh, self.f).transpose(1, 2)
        q, k, v = sp(self.to_q(h)), sp(self.to_k(h)), sp(self.to_v(h))
        eta = self.eta_base * torch.sigmoid(self.to_lr(h)).transpose(1, 2) / self.f      # (B, nh, T)
        if state is None:
            state = self.init_state(B)
        block = ttt_linear_block if self.kind == "linear" else ttt_mlp_block
        outs, losses = [], []
        snaps = [flat_state(state).detach()] if want_trace else None
        for s in range(0, T, self.b):
            e = min(s + self.b, T)
            o, state, l = block(state, q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], eta[:, :, s:e], self.ln_w, self.ln_b)
            outs.append(o)
            losses.append(l)
            if want_trace:
                snaps.append(flat_state(state).detach())
        z = torch.cat(outs, 2).transpose(1, 2).reshape(B, T, -1)
        read = self.merge(z)
        info = None
        if want_trace:
            info = dict(surprise=torch.cat(losses, 2).detach().float().mean((0, 1)),
                        lr=eta.detach().float().mean((0, 1)),
                        snaps=torch.stack(snaps, dim=1))
        return read, state, info
