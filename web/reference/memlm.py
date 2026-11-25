"""Lõi suy luận tham chiếu cho web app (không dùng biến toàn cục, không phụ thuộc notebook).

Tương thích checkpoint do notebook sinh ra: {"mem","out","gate","step","best","cfg"}.
Chế độ: "window" (backbone thuần, chỉ thấy truy vấn), "full" (backbone thấy cả ngữ cảnh, cận trên),
        "titans" | "ttt_linear" | "ttt_mlp" (backbone + memory đã nạp tài liệu).
Trạng thái memory là hàm thuần: ingest() trả về MemState MỚI, truy vấn không bao giờ sửa MemState.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
from ttt_core import TTTMem


@dataclass
class AdapterCfg:
    kind: str
    model_name: str = "Qwen/Qwen2.5-0.5B"
    seg_len: int = 256
    mem_chunk: int = 16
    mem_heads: int = 4
    mem_dim_head: int = 64
    mem_depth: int = 2
    use_momentum: bool = True
    titans_inner_lr_cap: float = 1.0
    layer: Optional[int] = None
    titans_kwargs: dict = field(default_factory=dict)   # dành cho v2 (init_decay_bias, max_lr, qk_rmsnorm...)

    @classmethod
    def from_ckpt(cls, blob):
        c = dict(blob["cfg"])
        known = {k: c[k] for k in cls.__dataclass_fields__ if k in c}
        return cls(**known)


@dataclass
class MemState:
    """Không trong suốt với UI; chỉ Engine được đọc. w_ref = trạng thái ban đầu (để tính ‖M_t−M_0‖)."""
    state: object
    w_ref: Optional[torch.Tensor]
    n_tokens: int = 0
    trace: list = field(default_factory=list)   # các điểm theo token để vẽ biểu đồ


def _flat_titans(st):
    return torch.cat([v.flatten(2) for k, v in st.updates.items() if k.startswith("model.weights")], dim=-1)


def _titans_memory_kwargs(cfg: AdapterCfg) -> dict:
    """Resolve saved v2 lr cap; legacy checkpoints retain the library default (1.0)."""
    kwargs = dict(cfg.titans_kwargs)
    kwargs["default_step_transform_max_lr"] = float(cfg.titans_inner_lr_cap)
    return kwargs


class MemAdapter(nn.Module):
    """Tên thuộc tính mem/out/gate trùng notebook để load_state_dict khớp."""

    def __init__(self, lm, cfg: AdapterCfg):
        super().__init__()
        self.lm, self.cfg = [lm], cfg                     # list: không đăng ký backbone vào state_dict
        d = lm.config.hidden_size
        if cfg.kind == "titans":
            from titans_pytorch import NeuralMemory
            self.mem = NeuralMemory(dim=d, chunk_size=cfg.mem_chunk, batch_size=cfg.mem_chunk, heads=cfg.mem_heads,
                                    dim_head=cfg.mem_dim_head, momentum=cfg.use_momentum,
                                    default_model_kwargs=dict(depth=cfg.mem_depth, expansion_factor=4.0),
                                    **_titans_memory_kwargs(cfg))
            for i, p in enumerate(self.mem.memory_model_parameters):
                self.mem.memory_model_parameters[i] = nn.Parameter(p.detach().clone().contiguous())
        else:
            self.mem = TTTMem(d, cfg.mem_heads, cfg.mem_dim_head, cfg.mem_chunk, "linear" if cfg.kind == "ttt_linear" else "mlp")
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.layer = cfg.layer if cfg.layer is not None else lm.config.num_hidden_layers // 2
        self._state, self._enabled, self._trace, self._w_ref, self._calls = None, False, False, None, []
        lm.model.layers[self.layer].register_forward_hook(self._hook)

    def _call(self, h):
        if self.cfg.kind == "titans":
            r, st, sur = self.mem(h, state=self._state, return_surprises=True)
            info = dict(surprise=sur[0].detach().float().mean((0, 1)), lr=sur[1].detach().float().mean((0, 1)),
                        snaps=_flat_titans(st).detach().float()) if self._trace else None
            return r, st, info
        return self.mem(h, state=self._state, want_trace=self._trace)

    def _hook(self, module, args, output):
        if not self._enabled:
            return output
        h = output[0] if isinstance(output, tuple) else output
        r, st, info = self._call(h.float())
        self._state = st
        if self._trace and info is not None:
            snaps = info["snaps"]
            if self._w_ref is None:
                self._w_ref = snaps[:, 0].clone()
            dabs = (snaps - self._w_ref[:, None]).norm(dim=-1)
            self._calls.append(dict(surprise=info["surprise"].cpu(), theta=info["lr"].cpu(),
                                    dist=(dabs / self._w_ref.norm(dim=-1, keepdim=True)).mean(0).cpu(),
                                    dist_abs=dabs.mean(0).cpu()))
        h2 = h + (self.gate * self.out(r)).to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2


class Engine:
    def __init__(self, lm, encode, decode, adapters: dict[str, MemAdapter], device="cpu"):
        self.lm, self.encode, self.decode, self.adapters, self.device = lm, encode, decode, adapters, device
        for p in lm.parameters():
            p.requires_grad_(False)
        lm.eval()
        self.seg_len = next(iter(adapters.values())).cfg.seg_len if adapters else 256
        self._ck0 = self.checksum()

    # ---- bất biến ----
    def checksum(self):
        h = hashlib.sha256()
        for p in self.lm.parameters():
            h.update(p.detach().float().sum().cpu().numpy().tobytes())
        return h.hexdigest()

    def backbone_unchanged(self):
        return self.checksum() == self._ck0

    def _set(self, name, state=None, enabled=True, trace=False):
        for n, a in self.adapters.items():
            a._enabled = enabled and n == name
        if name in self.adapters:
            a = self.adapters[name]
            a._state, a._trace = (state.state if state else None), trace
            a._w_ref, a._calls = (state.w_ref if state else None), []
        return self.adapters.get(name)

    # ---- API ----
    def new_state(self, kind):
        return MemState(None, None, 0)

    @torch.no_grad()
    def ingest(self, kind, text_or_ids, state: Optional[MemState] = None, trace=False) -> MemState:
        ids = text_or_ids if torch.is_tensor(text_or_ids) else torch.tensor(self.encode(text_or_ids), dtype=torch.long)
        state = state or self.new_state(kind)
        a = self._set(kind, state, True, trace)
        base = state.n_tokens
        tr = list(state.trace)
        for s in range(0, len(ids), self.seg_len):
            seg = ids[s:s + self.seg_len][None].to(self.device)
            self.lm(input_ids=seg, use_cache=False)
            for c in a._calls:
                for i in range(len(c["dist"]) - 1):
                    tr.append(dict(block=len(tr), dist=float(c["dist"][i + 1]), dist_abs=float(c["dist_abs"][i + 1])))
                tr_s = c["surprise"].tolist()
                tr.append(dict(surprise_tokens=tr_s, theta=c["theta"].tolist()))
            a._calls = []
        out = MemState(a._state, a._w_ref if a._w_ref is not None else state.w_ref, base + len(ids), tr if trace else [])
        for ad in self.adapters.values():
            ad._enabled = False
        return out

    @torch.no_grad()
    def token_logprobs(self, mode, prompt_ids, answer_ids, state: Optional[MemState] = None, context_ids=None):
        pr, an = (torch.as_tensor(x, dtype=torch.long) for x in (prompt_ids, answer_ids))
        assert len(pr) + len(an) <= self.seg_len, "truy vấn phải nằm trong 1 segment"
        if mode == "full":
            seq = torch.cat([torch.as_tensor(context_ids, dtype=torch.long)[-(self.seg_len * 8):], pr, an])
            self._set(None, enabled=False)
        elif mode == "window":
            seq = torch.cat([pr, an]); self._set(None, enabled=False)
        else:
            seq = torch.cat([pr, an]); self._set(mode, state, True, False)
        n = len(an)
        logits = self.lm(input_ids=seq[None].to(self.device), use_cache=False, logits_to_keep=n + 1).logits[0].float()
        for ad in self.adapters.values():
            ad._enabled = False
        logp = F.log_softmax(logits[-n - 1:-1], dim=-1)
        return logp[torch.arange(n), an.to(self.device)].cpu()

    def score(self, mode, prompt, candidates, state=None, context_ids=None, mean=False):
        """Xếp hạng ứng viên theo logP (tổng, hoặc trung bình theo token nếu mean=True)."""
        pi = self.encode(prompt)
        rows = []
        for c in candidates:
            lp = self.token_logprobs(mode, pi, self.encode(c), state, context_ids)
            rows.append(dict(candidate=c, logp=float(lp.sum()), logp_mean=float(lp.mean()), n_tokens=len(lp)))
        rows.sort(key=lambda r: -(r["logp_mean"] if mean else r["logp"]))
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return rows

    @torch.no_grad()
    def generate(self, mode, prompt_ids, state=None, context_ids=None, max_new_tokens=64):
        """Autoregressively generate an answer; the ingested memory state stays immutable."""
        if not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        pr = torch.as_tensor(prompt_ids, dtype=torch.long)
        if len(pr) == 0:
            raise ValueError("prompt must not be empty")
        if mode == "full":
            ctx = torch.as_tensor(context_ids if context_ids is not None else [], dtype=torch.long)[-(self.seg_len * 8):]
            seq = torch.cat([ctx, pr])
            self._set(None, enabled=False)
        elif mode == "window":
            seq = pr
            self._set(None, enabled=False)
        else:
            if mode not in self.adapters:
                raise ValueError(f"Unknown adapter: {mode}")
            seq = pr
            self._set(mode, state, True, False)
        try:
            result = self.lm.generate(input_ids=seq[None].to(self.device), max_new_tokens=max_new_tokens,
                                      do_sample=False, use_cache=True,
                                      pad_token_id=getattr(self.lm.config, "eos_token_id", None))
            return result[0, len(seq):].detach().cpu().tolist()
        finally:
            for adapter in self.adapters.values():
                adapter._enabled = False


def load_adapter(lm, ckpt_path, device="cpu"):
    blob = torch.load(ckpt_path, map_location=device)
    cfg = AdapterCfg.from_ckpt(blob)
    a = MemAdapter(lm, cfg).to(device)
    a.mem.load_state_dict(blob["mem"]); a.out.load_state_dict(blob["out"])
    a.gate.data = blob["gate"].to(device) if torch.is_tensor(blob["gate"]) else torch.tensor(float(blob["gate"]), device=device)
    return a, cfg
