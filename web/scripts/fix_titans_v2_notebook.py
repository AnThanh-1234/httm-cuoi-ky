"""Final consistency fixes for the one-model Kaggle Titans v2 notebook."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[2] / ".kaggle-demo" / "titans-factmix-v2" / "titans_v2_factmix.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))

def get(index):
    return "".join(notebook["cells"][index]["source"])

def put(index, text):
    notebook["cells"][index]["source"] = text.splitlines(keepends=True)

# This run starts from scratch. Any explicitly supplied resume path must exist
# and must carry the same inner-loop learning-rate cap.
settings = get(2)
settings = settings.replace(
    'INIT_CKPT = os.environ.get("INIT_CKPT", "/kaggle/input/titans-vi-memory-resume-2665/last.pt")        # đường dẫn last.pt từ phiên trước (nếu muốn tiếp tục)',
    'INIT_CKPT = None  # v2 train từ đầu; không resume checkpoint v1',
)
put(2, settings)

training = get(12)
old_resume = '''start_step, best_val = 0, float("inf")
_resume = INIT_CKPT or os.path.join(CKPT_DIR, "last.pt")
if os.path.exists(_resume):
    start_step, best_val = load_ckpt(_resume)
    print("Đã resume từ", _resume, "| bước", start_step, "| best", best_val)
else:
    print("Bắt đầu từ đầu.")'''
new_resume = '''start_step, best_val = 0, float("inf")
if INIT_CKPT:
    if not os.path.isfile(INIT_CKPT):
        raise FileNotFoundError(f"INIT_CKPT được đặt nhưng không tồn tại: {INIT_CKPT}")
    resume_blob = torch.load(INIT_CKPT, map_location="cpu")
    resume_cfg = resume_blob.get("cfg", {})
    saved_cap = resume_cfg.get("titans_inner_lr_cap")
    if saved_cap is None or abs(float(saved_cap) - TITANS_INNER_LR_CAP) > 1e-12:
        raise ValueError(f"Không resume: checkpoint lr cap={saved_cap}, cấu hình hiện tại={TITANS_INNER_LR_CAP}")
    start_step, best_val = load_ckpt(INIT_CKPT)
    print("Đã resume từ", INIT_CKPT, "| bước", start_step, "| best", best_val)
else:
    print("Train v2 từ đầu; không resume checkpoint v1.")'''
if old_resume not in training:
    raise RuntimeError("Không tìm thấy block resume dự kiến trong cell 12")
training = training.replace(old_resume, new_resume)
put(12, training)

loop = get(14)
old_tail = '''save_ckpt("last.pt", step, best_val)
save_ckpt(f"{MEMORY_KIND}_vi_memory.pt", step, best_val, with_opt=False)
model.mem.eval(); model.out.eval()
print("Đã lưu checkpoint tại", CKPT_DIR, "| bước", step)'''
new_tail = '''save_ckpt("last.pt", step, best_val)
save_ckpt(f"{MEMORY_KIND}_vi_memory.pt", step, best_val, with_opt=False)
model.mem.eval(); model.out.eval()
final_blob = torch.load(os.path.join(CKPT_DIR, f"{MEMORY_KIND}_vi_memory.pt"), map_location="cpu")
torch.save(final_blob, "/kaggle/working/titans_v2.pt")
print("Đã lưu checkpoint /kaggle/working/titans_v2.pt | bước", step)
print("Titans init:", TITANS_KWARGS, "| outer gate:", float(model.gate.detach().cpu()))'''
if old_tail not in loop:
    raise RuntimeError("Không tìm thấy block lưu checkpoint dự kiến trong cell 14")
loop = loop.replace(old_tail, new_tail)
put(14, loop)

# Cell 22 defines the scoring helper only; the final cell evaluates the fixed
# held-out facts and does not run a TTT comparison.
helper = '''@torch.no_grad()
def answer_score(model, context_ids, prompt_ids, answer_ids, mode):
    ans = torch.tensor(answer_ids, device=device)
    pr = torch.tensor(prompt_ids, device=device)
    if mode == "memory":
        model.reset(); model.enabled = True
        for s in range(0, len(context_ids), SEG_LEN):
            model.lm(input_ids=context_ids[s:s + SEG_LEN][None].to(device), use_cache=False)
        seq = torch.cat([pr, ans])
    elif mode == "window":
        model.enabled = False; model.reset()
        seq = torch.cat([pr, ans])
    else:
        model.enabled = False; model.reset()
        seq = torch.cat([context_ids.to(device), pr, ans])
    n = len(answer_ids)
    logits = model.lm(input_ids=seq[None], use_cache=False, logits_to_keep=n + 1).logits[0].float()
    model.enabled = True
    logp = F.log_softmax(logits[-n - 1:-1], dim=-1)
    tok_lp = logp[torch.arange(n), ans]
    top1 = (logp.argmax(-1) == ans).all().item()
    return float(tok_lp.sum()), float(tok_lp.exp().prod()), bool(top1)
'''
put(22, helper)

path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {path}")
