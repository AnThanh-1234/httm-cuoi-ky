"""Free-form decoding keeps the ingested memory state available for repeat queries."""
import os
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memlm import Engine, MemState

class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(eos_token_id=2)
        self.seen = None
    def generate(self, input_ids, **kwargs):
        self.seen = (input_ids.detach().cpu().tolist(), kwargs)
        extra = torch.tensor([[31, 32]], device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)

def test_generate_uses_ingested_state_and_disables_hook_afterwards():
    lm = TinyLM()
    adapter = SimpleNamespace(_enabled=False, cfg=SimpleNamespace(seg_len=256))
    eng = Engine(lm, lambda _: [], lambda ids: "", {"titans": adapter})
    original_state = object()
    state = MemState(original_state, None, 7)
    out = eng.generate("titans", [10, 11], state, max_new_tokens=2)
    assert out == [31, 32]
    assert adapter._state is original_state
    assert adapter._enabled is False
    assert lm.seen[0] == [[10, 11]]
    assert lm.seen[1]["do_sample"] is False
    assert state.state is original_state

def test_generate_rejects_unbounded_length():
    eng = Engine(TinyLM(), lambda _: [], lambda ids: "", {})
    try:
        eng.generate("window", [1], max_new_tokens=129)
    except ValueError as exc:
        assert "between 1 and 128" in str(exc)
    else:
        raise AssertionError("generation length should be bounded")
