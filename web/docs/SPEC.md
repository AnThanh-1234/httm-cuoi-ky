# SPEC — Web app truy xuất: Titans vs TTT vs baseline (tiếng Việt)

## 0. Mục tiêu và nguyên tắc trung thực
Cho giảng viên **nạp một tài liệu**, **đặt câu hỏi**, và **tự kiểm chứng** câu trả lời bằng cách so các mô hình trên cùng dữ liệu:
`window` (Qwen thuần, chỉ thấy câu hỏi) · `full` (Qwen thấy cả tài liệu, cận trên) · `titans` · `ttt_linear` (· `ttt_mlp` nếu có).

Quy tắc không được vi phạm:
1. **Không hard-code kết quả**. Mọi số liệu hiển thị đến từ lần chạy thật hoặc từ `docs/kaggle_results.json` (gắn nhãn "từ log Kaggle").
2. **Đáp án đúng do người kiểm tra cung cấp** (ground truth), app không "tự biết" đúng/sai.
3. Hiện trạng thực tế (xem §1): **memory hiện tại chưa truy xuất được sự kiện bịa**. UI phải hiển thị cảnh báo này (banner) cho tới khi checkpoint v2 qua được ngưỡng ở §8. Không được che bằng cách chọn ví dụ dễ.
4. Đây là *memory adapter trên Qwen đóng băng*, **không** phải Titans/TTT end-to-end. Dùng đúng tên này trong UI.
5. Bất biến: tham số backbone không đổi (checksum), truy vấn không làm thay đổi trạng thái phiên.

## 1. Số liệu nền (từ log Kaggle, xem docs/kaggle_results.json)
| | base | reset | memory | lợi ích (reset−memory) |
|---|---|---|---|---|
| Titans | 2.9461 | 2.8926 | 2.8731 | +0.0195 |
| TTT-Linear | 2.9461 | 2.9022 | 2.8390 | +0.0632 |
- TTT nhanh gấp ~2× (84.7 vs 171.6 ms/segment) và fast-weight nhỏ hơn 8× (0.07 vs 0.52 MB).
- Truy xuất 12 sự kiện bịa ở khoảng cách 0–7 segment: window/Titans/TTT đều top-1 = 0/12, logP ≈ −11; full-context 12/12, logP ≈ −0.8.
- **Diễn giải đúng**: TTT > Titans ở bài toán *giảm loss ngôn ngữ trên Wikipedia*, nhưng khác biệt có thể do (a) forgetting gate mặc định của Titans ≈ 0.51/chunk (dịch chuyển trọng số = 1.000 phù hợp với giả thuyết này nhưng CHƯA kiểm chứng trên gate đã train), (b) mục tiêu train không thưởng cho recall chính xác. Không kết luận "TTT tốt hơn Titans" cho truy xuất. Phase 0 (docs/PHASE0_retrain.md) sửa việc này.

## 2. Kiến trúc
```
repo/
  backend/        FastAPI (Python 3.11+), import reference/memlm.py và reference/ttt_core.py (copy nguyên, không viết lại)
  frontend/       Vite + React + TypeScript, Vietnamese UI, chart: uPlot hoặc Recharts, không CDN (chạy offline)
  models/         checkpoint (gitignore)
  data/examples/  vài tài liệu mẫu tiếng Việt do người dùng sửa được
  tests/          pytest backend + Playwright smoke
```
Một lệnh chạy: `make dev` (backend :8000, frontend :5173); `make build` cho bản tĩnh phục vụ qua backend.
Biến môi trường: `DEVICE=cuda|cpu`, `MOCK=1`, `MODELS_DIR`, `BACKBONE=Qwen/Qwen2.5-0.5B`.
Một GPU: dùng khóa (asyncio.Lock) để chỉ một job suy luận chạy một lúc; job dài chạy nền, báo tiến độ qua SSE.

## 3. Mô hình dữ liệu
- `Session{id, created, doc_text, doc_tokens, states:{kind: MemState}, notes}` — giữ trong RAM, lưu JSON (không lưu state tensor, nạp lại bằng cách ingest lại, có seed cố định).
- `MemState` là đối tượng của `memlm.py` (opaque). Truy vấn **không** được sửa nó.
- `Question{prompt, candidates[], gold_index}`; `gold_index` do người kiểm tra nhập.
- `RunResult{mode, ranking[{candidate, logp, logp_mean, rank}], gold_rank, top1_correct, latency_ms}`.

## 4. API (JSON, tiếng Việt ở message người dùng)
| Method | Path | Mô tả |
|---|---|---|
| GET | /api/health | thiết bị, backbone, checksum, danh sách checkpoint, `mock` |
| GET | /api/models | các chế độ khả dụng + cfg + nhãn "v1/v2" + `passed_retrieval_gate` |
| POST | /api/sessions | tạo phiên `{doc_text}` → `{id, n_tokens}` |
| POST | /api/sessions/{id}/ingest | `{kinds:[...], trace:bool}` nạp tài liệu (SSE tiến độ theo segment) |
| POST | /api/sessions/{id}/score | `{prompt, candidates[], gold_index?, modes[], mean:bool}` → xếp hạng từng mode |
| POST | /api/sessions/{id}/generate | `{prompt, modes[], max_new_tokens}` sinh tự do (chỉ để minh họa, ghi rõ KHÔNG dùng để chấm) |
| GET | /api/sessions/{id}/diagnostics | trace: ‖M_t−M_0‖ (tuyệt đối + chuẩn hóa), surprise, θ/lr theo token, gate α/η nếu Titans |
| POST | /api/benchmark | `{items:[{doc, prompt, candidates, gold}], distances:[0,1,2,4,7], modes, seeds}` → bảng + CI |
| GET | /api/benchmark/{job} | trạng thái/kết quả; export CSV/JSON |
| GET | /api/reference/kaggle | trả docs/kaggle_results.json |
Lỗi rõ ràng: truy vấn dài hơn 1 segment (256 token) → 422 kèm số token; tài liệu rỗng; checkpoint thiếu.

## 5. Các trang
1. **Tổng quan** — giải thích 1 màn hình: backbone đóng băng + memory học lúc test; sơ đồ; bảng §1 (nhãn "log Kaggle"); banner trung thực (§0.3); so sánh chi phí (KV cache tăng tuyến tính vs fast-weight cố định, số liệu thật).
2. **Memory Lab** — nạp văn bản, chạy `ingest` có trace; vẽ **đường trọng số thay đổi theo token** cho Titans và TTT trên cùng trục (hai thang: tuyệt đối & chuẩn hóa, chú thích rằng không so trực tiếp), surprise theo token, θ/lr; đường "Qwen thuần" là đường phẳng 0 (không có trạng thái) kèm chú thích. Heatmap token: gain = logP(memory) − logP(window) trên văn bản người dùng dán (đây là "lợi ích memory" theo token, tô màu xanh/đỏ, có thanh trượt ngưỡng).
3. **Kiểm tra truy xuất** (trang chính cho thầy) — 3 bước: (a) dán tài liệu hoặc chọn mẫu; (b) nhập câu hỏi + **2–6 đáp án ứng viên** và đánh dấu đáp án đúng (ground truth); (c) bấm chạy → bảng hạng của window / full / titans / ttt, màu xanh nếu đáp án đúng đứng hạng 1, hiện logP từng ứng viên, độ chênh so với window. Nút "Đổi vị trí sự kiện" chèn N đoạn nhiễu giữa sự kiện và câu hỏi (khoảng cách 0/1/2/4/7 segment). Nhận biết rò rỉ: cảnh báo nếu đáp án xuất hiện trong prompt.
4. **Benchmark** — chạy hàng loạt (mặc định: 12 sự kiện bịa × khoảng cách × seed), hiển thị top-1 accuracy, MRR, logP trung bình **kèm khoảng tin cậy bootstrap 95%**, đường theo khoảng cách. Có nút "Tái hiện số trong log Kaggle" để đối chiếu.
5. **Kết quả Kaggle** — bảng và đồ thị từ `kaggle_results.json`, gắn nguồn/ngày.
6. **Giới hạn & cách đọc** — danh sách giới hạn trung thực (§0, §1).

## 6. Quy tắc tính toán (bắt buộc)
- Chấm theo `logp` tổng của ứng viên; tùy chọn `mean` (chuẩn hóa độ dài). Mặc định hiển thị cả hai, hạng chính theo `mean` khi độ dài ứng viên khác nhau.
- Mỗi mode dùng **đúng cùng** prompt và ứng viên. `full` cắt ngữ cảnh từ bên trái nếu quá dài và hiển thị cảnh báo.
- Ứng viên gồm đáp án đúng + distractor cùng định dạng (số/tên/ngày). Có công cụ sinh distractor ngẫu nhiên có seed; hiển thị seed.
- Số thập phân Việt "7,31" được tách token như chuỗi; không chuẩn hóa ngầm.
- Không dùng temperature/sampling cho bất cứ phép chấm nào.

## 7. UX và xuất dữ liệu
- Mọi bảng/đồ thị có nút xuất CSV/PNG; "Xuất báo cáo phiên" = JSON gồm tài liệu, câu hỏi, ứng viên, gold, kết quả, cfg checkpoint, checksum backbone, git sha, seed, thời gian.
- Chế độ trình chiếu (chữ lớn), dark/light.
- Mọi con số có tooltip nói nguồn (chạy trực tiếp / log Kaggle).
- Khi `MOCK=1`: banner đỏ cố định "DỮ LIỆU MÔ PHỎNG — không phải kết quả thật".

## 8. Cổng truy xuất (retrieval gate) — khi nào được bỏ banner
Dùng benchmark nội bộ trên checkpoint v2 với sự kiện bịa **chưa thấy lúc train**, 2 seed: top-1 trong tập 4 ứng viên phải cao hơn `window` có ý nghĩa (CI bootstrap 95% không chứa 0) ở khoảng cách ≥ 2 segment. Nếu không đạt, giữ banner "memory chưa truy xuất được" và `passed_retrieval_gate=false`. Cờ này do backend tính từ benchmark, không do người sửa tay.

## 9. Tiêu chí nghiệm thu
1. `pytest reference/tests` và `pytest tests` xanh (có `DEBUG_CKPT_DIR` hoặc `MOCK=1`).
2. Điều kiện tái hiện: với checkpoint v1 + Qwen thật, benchmark ở khoảng cách 7 trả `window ≈ −11.8, titans ≈ −11.6, ttt_linear ≈ −11.3, full ≈ −0.8` (sai số ±0.5; do tập sự kiện/seeds có thể khác, ghi rõ).
3. Truy vấn lặp lại cho kết quả giống hệt; truy vấn không đổi trạng thái; checksum backbone không đổi sau 100 truy vấn.
4. Cùng một phiên: ingest theo một lần == ingest theo hai nửa (xem test).
5. Banner trung thực hiển thị khi `passed_retrieval_gate=false`; `MOCK` luôn có nhãn.
6. Không có số kết quả nào nằm trong mã frontend (grep kiểm tra) ngoài file nhập từ `/api/reference/kaggle`.
7. Chạy được trên máy không có GPU (chậm hơn) và offline sau khi đã cache Qwen.
8. Playwright smoke: tạo phiên → ingest → chạy truy xuất → thấy bảng hạng 4 mode → xuất JSON.

## 10. Ngoài phạm vi (đừng làm)
Huấn luyện trong web; đăng nhập/nhiều người dùng; claim "Titans/TTT thật"; chat tự do được trình bày như bằng chứng.
