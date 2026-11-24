"""Bất biến lõi. Chạy: DEBUG_CKPT_DIR=<thư mục chứa {titans,ttt_linear}_vi_memory.pt> pytest reference/tests -q
Backbone DEBUG = Qwen2 ngẫu nhiên nhỏ (seed 1234) giống DEBUG của notebook, nên không cần Internet."""
import os, sys, glob, torch, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import Qwen2Config, Qwen2ForCausalLM
from memlm import Engine, load_adapter

CK = os.environ.get("DEBUG_CKPT_DIR", "")
KINDS = [k for k in ("titans", "ttt_linear") if os.path.exists(os.path.join(CK, f"{k}_vi_memory.pt"))]
pytestmark = pytest.mark.skipif(not KINDS, reason="cần DEBUG_CKPT_DIR")
enc = lambda t: [b % 500 for b in t.encode("utf-8")]
dec = lambda ids: "".join(chr(32 + (int(i) % 90)) for i in ids)


@pytest.fixture(scope="module")
def eng():
    torch.manual_seed(1234)
    lm = Qwen2ForCausalLM(Qwen2Config(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)).eval()
    ads = {k: load_adapter(lm, os.path.join(CK, f"{k}_vi_memory.pt"))[0] for k in KINDS}
    return Engine(lm, enc, dec, ads)


DOC = "Theo ghi chep cua phong thi nghiem Delta, dinh luat Zorvan phat bieu rang he so bang 7,31 don vi. " * 3


@pytest.mark.parametrize("kind", KINDS)
def test_query_does_not_mutate_state(eng, kind):
    st = eng.ingest(kind, DOC)
    before = [v.clone() for v in (st.state.weights.values() if kind == "titans" and hasattr(st.state, "weights") else st.state.values())] if kind != "titans" else None
    a = eng.score(kind, "Hệ số bằng", [" 7,31", " 9,99"], st)
    b = eng.score(kind, "Hệ số bằng", [" 7,31", " 9,99"], st)
    assert [r["logp"] for r in a] == [r["logp"] for r in b]          # truy vấn lặp lại -> y hệt
    if before is not None:
        for x, y in zip(before, st.state.values()):
            assert torch.equal(x, y)


@pytest.mark.parametrize("kind", KINDS)
def test_state_carry_equals_one_shot(eng, kind):
    ids = torch.tensor(enc(DOC))
    cut = eng.seg_len
    one = eng.ingest(kind, ids)
    two = eng.ingest(kind, ids[cut:], eng.ingest(kind, ids[:cut]))
    a = eng.token_logprobs(kind, enc("Hệ số bằng"), enc(" 7,31"), one)
    b = eng.token_logprobs(kind, enc("Hệ số bằng"), enc(" 7,31"), two)
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_window_independent_of_state(eng, kind):
    a = eng.token_logprobs("window", enc("Hệ số bằng"), enc(" 7,31"))
    eng.ingest(kind, DOC)
    b = eng.token_logprobs("window", enc("Hệ số bằng"), enc(" 7,31"))
    assert torch.equal(a, b)


@pytest.mark.parametrize("kind", KINDS)
def test_memory_changes_output_and_backbone_frozen(eng, kind):
    st = eng.ingest(kind, DOC)
    w = eng.token_logprobs("window", enc("Hệ số bằng"), enc(" 7,31"))
    m = eng.token_logprobs(kind, enc("Hệ số bằng"), enc(" 7,31"), st)
    assert not torch.equal(w, m)
    assert eng.backbone_unchanged()


@pytest.mark.parametrize("kind", KINDS)
def test_causality(eng, kind):
    # đổi token cuối của tài liệu không được làm đổi log-prob các token trước đó (kiểm tra qua trace dist)
    ids = torch.tensor(enc(DOC)[: eng.seg_len])
    ids2 = ids.clone(); ids2[-1] = (ids2[-1] + 7) % 500
    a = eng.ingest(kind, ids, trace=True); b = eng.ingest(kind, ids2, trace=True)
    da = [t["dist"] for t in a.trace if "dist" in t][:-1]
    db = [t["dist"] for t in b.trace if "dist" in t][:-1]
    assert da and all(abs(x - y) < 1e-6 for x, y in zip(da, db))
