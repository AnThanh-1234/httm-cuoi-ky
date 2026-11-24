"""FastAPI service for the retrieval verification app.

Real adapter loading is intentionally isolated behind Engine; MOCK mode is an
explicit deterministic UI/API simulator and must never be read as model output.
"""
from __future__ import annotations
import asyncio, csv, hashlib, io, json, os, random, time, uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MOCK = os.getenv("MOCK", "0").lower() in {"1", "true", "yes"}
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT / "models")))
app = FastAPI(title="Kiểm tra truy xuất memory test-time", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
SESSIONS: dict[str, dict[str, Any]] = {}
JOBS: dict[str, dict[str, Any]] = {}
GPU_LOCK = asyncio.Lock()
_ENGINE = None
PASSED_RETRIEVAL_GATE = False
KINDS = ["titans", "ttt_linear", "ttt_mlp"]
MODES = ["window", "full", "titans", "ttt_linear", "ttt_mlp"]

class SessionIn(BaseModel): doc_text: str
class IngestIn(BaseModel): kinds: list[str] = Field(default_factory=lambda: ["titans", "ttt_linear"]); trace: bool = True
class ScoreIn(BaseModel): prompt: str; candidates: list[str]; gold_index: int | None = None; modes: list[str] = Field(default_factory=lambda: MODES[:4]); mean: bool = True
class GenerateIn(BaseModel): prompt: str; modes: list[str] = MODES[:4]; max_new_tokens: int = 32
class BenchmarkIn(BaseModel): items: list[dict] = Field(default_factory=list); distances: list[int] = Field(default_factory=lambda: [0,1,2,4,7]); modes: list[str] = MODES[:4]; seeds: list[int] = Field(default_factory=lambda: [1,2])

def tokens(text: str) -> list[str]: return text.split()

def get_engine():
    global _ENGINE
    if _ENGINE is not None: return _ENGINE
    if MOCK: return None
    try:
        import sys, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, str(ROOT / "reference"))
        from memlm import Engine, load_adapter
        device=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        name=os.getenv("BACKBONE", "Qwen/Qwen2.5-0.5B")
        tok=AutoTokenizer.from_pretrained(name, local_files_only=os.getenv("OFFLINE", "0")=="1")
        lm=AutoModelForCausalLM.from_pretrained(name, torch_dtype=(torch.float16 if device.startswith("cuda") else torch.float32), local_files_only=os.getenv("OFFLINE", "0")=="1").to(device).eval()
        adapters={}
        ckpt_names={"titans":"titans_v2.pt" if (MODELS_DIR/"titans_v2.pt").exists() else "titans_vi_memory.pt", "ttt_linear":"ttt_linear_v2.pt" if (MODELS_DIR/"ttt_linear_v2.pt").exists() else "ttt_linear_vi_memory.pt", "ttt_mlp":"ttt_mlp_v2.pt"}
        for kind,file in ckpt_names.items():
            p=MODELS_DIR/file
            if p.exists(): adapters[kind]=load_adapter(lm,str(p),device)[0]
        _ENGINE={"engine":Engine(lm,lambda t:tok.encode(t,add_special_tokens=False),lambda ids:tok.decode(ids),adapters,device),"tokenizer":tok,"torch":torch,"device":device,"name":name}
        return _ENGINE
    except Exception as e:
        raise HTTPException(503,f"Không khởi tạo được Qwen/checkpoint: {type(e).__name__}: {e}") from e
def _mock_rows(sid: str, prompt: str, candidates: list[str], mode: str, mean: bool) -> list[dict]:
    # Stable random scores keep the mock repeatable without implying model quality.
    rows=[]
    for idx,c in enumerate(candidates):
        digest=hashlib.sha256(f"{sid}|{prompt}|{c}|{mode}".encode()).digest()
        score=-0.25-(int.from_bytes(digest[:4],"big")/0xffffffff)*8
        n=max(1,len(tokens(c)))
        rows.append({"candidate":c,"logp":score*n,"logp_mean":score,"n_tokens":n})
    rows.sort(key=lambda r: -(r["logp_mean"] if mean else r["logp"]))
    for i,row in enumerate(rows): row["rank"]=i+1
    return rows

@app.get("/api/health")
def health():
    ckpts=[p.name for p in MODELS_DIR.glob("*.pt")] if MODELS_DIR.exists() else []
    checksum=_ENGINE["engine"].checksum() if _ENGINE else None
    return {"device":_ENGINE["device"] if _ENGINE else os.getenv("DEVICE","cpu"),"backbone":_ENGINE["name"] if _ENGINE else os.getenv("BACKBONE","Qwen/Qwen2.5-0.5B"),"checksum":checksum,"checkpoints":ckpts,"mock":MOCK}

@app.get("/api/models")
def models():
    names=[p.name for p in MODELS_DIR.glob("*.pt")] if MODELS_DIR.exists() else []
    checkpoint_names = {
        "titans": ("titans_v2.pt", "titans_vi_memory.pt"),
        "ttt_linear": ("ttt_linear_v2.pt", "ttt_linear_vi_memory.pt"),
        "ttt_mlp": ("ttt_mlp_v2.pt", "ttt_mlp_vi_memory.pt"),
    }
    mode_rows = []
    for mode in MODES:
        if mode in {"window", "full"}:
            available, label = True, "backbone"
        else:
            v2_name, v1_name = checkpoint_names[mode]
            available = MOCK or v2_name in names or v1_name in names
            label = "MOCK" if MOCK else ("v2" if v2_name in names else "v1" if v1_name in names else "không có checkpoint")
        mode_rows.append({"id": mode, "available": available, "label": label, "cfg": None})
    return {"modes": mode_rows, "passed_retrieval_gate": PASSED_RETRIEVAL_GATE, "mock": MOCK}

@app.post("/api/sessions")
def create_session(body: SessionIn):
    if not body.doc_text.strip(): raise HTTPException(422,"Tài liệu không được để trống.")
    sid=str(uuid.uuid4()); SESSIONS[sid]={"id":sid,"created":time.time(),"doc_text":body.doc_text,"doc_tokens":tokens(body.doc_text),"states":{},"trace":[]}
    return {"id":sid,"n_tokens":len(SESSIONS[sid]["doc_tokens"])}

@app.post("/api/sessions/{sid}/ingest")
async def ingest(sid: str, body: IngestIn):
    s=SESSIONS.get(sid)
    if not s: raise HTTPException(404,"Không tìm thấy phiên.")
    if any(k not in KINDS for k in body.kinds): raise HTTPException(422,"Mode ingest không hợp lệ.")
    async def events():
        total=max(1,(len(s['doc_tokens'])+255)//256)
        async with GPU_LOCK:
            if not MOCK:
                loaded=get_engine(); eng=loaded['engine']; ids=loaded['tokenizer'].encode(s['doc_text'],add_special_tokens=False)
                total=max(1,(len(ids)+eng.seg_len-1)//eng.seg_len)
                for kind in body.kinds:
                    if kind not in eng.adapters: raise HTTPException(409,f"Thiếu checkpoint cho {kind} trong {MODELS_DIR}.")
                    state=None
                    for i in range(total):
                        state=eng.ingest(kind,loaded['torch'].as_tensor(ids[i*eng.seg_len:(i+1)*eng.seg_len],dtype=loaded['torch'].long),state,trace=body.trace)
                        yield f"data: {json.dumps({'progress':(i+1)/total,'segment':i+1,'total':total,'mode':kind},ensure_ascii=False)}\n\n"
                    s['states'][kind]=state
                    s['trace']=state.trace
            else:
                for i in range(total):
                    await asyncio.sleep(0)
                    yield f"data: {json.dumps({'progress':(i+1)/total,'segment':i+1,'total':total},ensure_ascii=False)}\n\n"
                    for kind in body.kinds: s['states'][kind]={"n_tokens":len(s['doc_tokens']),"seed":1234}
                s['trace']=[]
            yield f"data: {json.dumps({'done':True,'kinds':body.kinds,'mock':MOCK},ensure_ascii=False)}\n\n"
    return StreamingResponse(events(),media_type="text/event-stream")

@app.post("/api/sessions/{sid}/score")
async def score(sid: str, body: ScoreIn):
    s=SESSIONS.get(sid)
    if not s: raise HTTPException(404,"Không tìm thấy phiên.")
    if not 2 <= len(body.candidates) <= 6: raise HTTPException(422,"Cần từ 2 đến 6 đáp án ứng viên.")
    if any(not c.strip() for c in body.candidates): raise HTTPException(422,"Đáp án ứng viên không được để trống.")
    if body.gold_index is not None and not 0 <= body.gold_index < len(body.candidates): raise HTTPException(422,"Đáp án đúng không hợp lệ.")
    if any(m not in MODES for m in body.modes): raise HTTPException(422,"Mode không hợp lệ.")
    loaded=get_engine() if not MOCK else None
    if MOCK:
        qlen=len(tokens(body.prompt))+max(map(lambda c:len(tokens(c)),body.candidates))
        if qlen>256: raise HTTPException(422,f"Truy vấn vượt 256 token (prompt: {len(tokens(body.prompt))}).")
    else:
        tok=loaded['tokenizer']; qlen=len(tok.encode(body.prompt,add_special_tokens=False))+max(len(tok.encode(c,add_special_tokens=False)) for c in body.candidates)
        if qlen>loaded['engine'].seg_len: raise HTTPException(422,f"Truy vấn vượt {loaded['engine'].seg_len} token (tổng prompt và đáp án dài nhất: {qlen}).")
    result={}
    async with GPU_LOCK:
        for mode in body.modes:
            if mode not in {"window","full"} and mode not in s["states"] and not MOCK:
                raise HTTPException(409,f"Hãy ingest tài liệu cho mode {mode} trước.")
            started=time.perf_counter()
            if MOCK:
                ranking=_mock_rows(sid,body.prompt,body.candidates,mode,body.mean)
            else:
                eng=loaded['engine']; tok=loaded['tokenizer']; prompt_ids=tok.encode(body.prompt,add_special_tokens=False)
                context=tok.encode(s['doc_text'],add_special_tokens=False)
                ranking=eng.score(mode,body.prompt,body.candidates,s['states'].get(mode),context_ids=context,mean=body.mean)
            result[mode]={"ranking":ranking,"gold_rank":next((r['rank'] for i,r in enumerate(ranking) if r['candidate']==body.candidates[body.gold_index]),None) if body.gold_index is not None else None,"top1_correct":bool(body.gold_index is not None and ranking[0]['candidate']==body.candidates[body.gold_index]),"latency_ms":(time.perf_counter()-started)*1000,"mock":MOCK}
    return {"session_id":sid,"results":result,"gold_index":body.gold_index,"mean_ranked":body.mean,"mock":MOCK}

@app.post("/api/sessions/{sid}/generate")
def generate(sid: str, body: GenerateIn):
    if sid not in SESSIONS: raise HTTPException(404,"Không tìm thấy phiên.")
    return {"warning":"Sinh tự do chỉ để minh họa, KHÔNG dùng để chấm.","mock":MOCK,"results":{m:"" for m in body.modes}}

@app.get("/api/sessions/{sid}/diagnostics")
def diagnostics(sid: str):
    if sid not in SESSIONS: raise HTTPException(404,"Không tìm thấy phiên.")
    return {"trace":SESSIONS[sid]["trace"],"mock":MOCK}

def _ci_mean(values: list[float], seed: int) -> list[float]:
    if not values: return [0.0,0.0]
    rng=random.Random(seed); means=[]
    for _ in range(1000): means.append(sum(rng.choices(values,k=len(values)))/len(values))
    means.sort(); return [means[24],means[974]]

async def _run_benchmark(jid: str, body: BenchmarkIn):
    global PASSED_RETRIEVAL_GATE
    job=JOBS[jid]; observations=[]
    try:
        loaded=get_engine() if not MOCK else None
        async with GPU_LOCK:
            for item_i,item in enumerate(body.items):
                doc=str(item.get("doc", "")); prompt=str(item.get("prompt", "")); candidates=item.get("candidates",[]); gold=item.get("gold")
                if not doc.strip() or not prompt or len(candidates)<2 or gold not in candidates: raise ValueError("Mỗi benchmark item cần doc, prompt, candidates và gold.")
                for distance in body.distances:
                    for seed in body.seeds:
                        noise=" ".join(["NHIỄU"]*(distance*256))
                        test_doc=(doc+" "+noise).strip()
                        states={}
                        if not MOCK:
                            eng=loaded['engine']; tok=loaded['tokenizer']; ids=tok.encode(test_doc,add_special_tokens=False)
                            for mode in body.modes:
                                if mode in eng.adapters: states[mode]=eng.ingest(mode,loaded['torch'].as_tensor(ids,dtype=loaded['torch'].long),trace=False)
                        sid=f"bench-{jid}-{item_i}-{distance}-{seed}"
                        for mode in body.modes:
                            if MOCK: rank=_mock_rows(sid,prompt,candidates,mode,True)
                            else: rank=loaded['engine'].score(mode,prompt,candidates,states.get(mode),context_ids=loaded['tokenizer'].encode(test_doc,add_special_tokens=False),mean=True)
                            gr=next(r['rank'] for r in rank if r['candidate']==gold)
                            observations.append({"item":item_i,"distance":distance,"seed":seed,"mode":mode,"gold_rank":gr,"top1":float(gr==1),"mrr":1/gr,"logp":next(r['logp'] for r in rank if r['candidate']==gold)})
            grouped=[]
            for distance in body.distances:
                for mode in body.modes:
                    subset=[r for r in observations if r['distance']==distance and r['mode']==mode]
                    if not subset: continue
                    accuracy=[r['top1'] for r in subset]; mrr=[r['mrr'] for r in subset]; lp=[r['logp'] for r in subset]
                    grouped.append({"mode":mode,"distance":distance,"n":len(subset),"top1_accuracy":sum(accuracy)/len(accuracy),"top1_ci95":_ci_mean(accuracy,101+distance),"mrr":sum(mrr)/len(mrr),"mrr_ci95":_ci_mean(mrr,203+distance),"mean_gold_logp":sum(lp)/len(lp),"logp_ci95":_ci_mean(lp,307+distance)})
            # Gate is backend-computed only from an explicitly designated held-out v2 evaluation.
            if not MOCK and body.items[0].get("gate_evaluation") and any("v2" in p.name for p in MODELS_DIR.glob("*.pt")) and len(body.seeds)>=2 and "window" in body.modes:
                passed=True
                for mode in ("titans","ttt_linear"):
                    if mode not in body.modes: passed=False; break
                    diffs=[]
                    for d in body.distances:
                        if d<2: continue
                        for idx in range(len(body.items)):
                            for seed in body.seeds:
                                m=next((o['top1'] for o in observations if o['item']==idx and o['distance']==d and o['seed']==seed and o['mode']==mode),None)
                                w=next((o['top1'] for o in observations if o['item']==idx and o['distance']==d and o['seed']==seed and o['mode']=='window'),None)
                                if m is not None and w is not None: diffs.append(m-w)
                    if not diffs or _ci_mean(diffs,900+len(diffs))[0] <= 0: passed=False
                PASSED_RETRIEVAL_GATE=passed
            job.update({"status":"completed","results":grouped,"observations":observations,"mock":MOCK,"created":job['created']})
    except Exception as e: job.update({"status":"failed","error":str(e),"mock":MOCK})

@app.post("/api/benchmark")
async def benchmark(body: BenchmarkIn):
    if not body.items: raise HTTPException(422,"Hãy cung cấp ít nhất một mục benchmark.")
    if any(d not in [0,1,2,4,7] for d in body.distances): raise HTTPException(422,"Khoảng cách hợp lệ: 0, 1, 2, 4, 7.")
    jid=str(uuid.uuid4()); JOBS[jid]={"job_id":jid,"status":"running","mock":MOCK,"results":[],"created":time.time()}
    asyncio.create_task(_run_benchmark(jid,body))
    return {"job_id":jid,"status":"running","mock":MOCK}
@app.get("/api/benchmark/{job}")
def get_benchmark(job: str, format: str="json"):
    j=JOBS.get(job)
    if not j: raise HTTPException(404,"Không tìm thấy benchmark.")
    if format=="csv":
        out=io.StringIO(); fields=["mode","distance","n","top1_accuracy","top1_ci95","mrr","mrr_ci95","mean_gold_logp","logp_ci95"]; w=csv.DictWriter(out,fieldnames=fields); w.writeheader(); w.writerows(j.get("results",[]))
        return StreamingResponse(iter([out.getvalue()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=benchmark.csv"})
    return j
@app.get("/api/reference/kaggle")
def kaggle():
    results = json.loads((ROOT/"docs"/"kaggle_results.json").read_text(encoding="utf-8"))
    v2_path = ROOT / "docs" / "eval_v2.json"
    if v2_path.exists():
        results["titans_v2_factmix"] = json.loads(v2_path.read_text(encoding="utf-8"))
    return results

STATIC=ROOT/"backend"/"static"
if STATIC.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets",StaticFiles(directory=STATIC/"assets"),name="assets")
    @app.get("/{path:path}")
    def frontend(path: str):
        target=STATIC/path
        return FileResponse(target if target.is_file() else STATIC/"index.html")

