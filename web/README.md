# Memory Lab — kiểm tra truy xuất

Ứng dụng web tiếng Việt để so sánh khả năng xếp hạng đáp án của Qwen thuần (`window`), Qwen với toàn bộ ngữ cảnh (`full`) và các memory adapter (`titans`, `ttt_linear`, tùy chọn `ttt_mlp`). Tên gọi luôn chỉ adapter trên backbone Qwen đóng băng, không phải Titans/TTT end-to-end.

## Chạy ứng dụng

Cần Python 3.11+, Node.js và npm. Tại thư mục `web/`:

```sh
uv venv .venv312 --python 3.12
# Windows
uv pip install --python .venv312/Scripts/python.exe -r backend/requirements.txt
# macOS/Linux: uv pip install --python .venv312/bin/python -r backend/requirements.txt
cd frontend && npm install && cd ..
```

Chạy giao diện và API cùng lúc:

```sh
MOCK=1 make dev
```

Trên PowerShell:

```powershell
$env:MOCK="1"
make dev
```

Vite chạy ở <http://localhost:5173>, FastAPI ở <http://localhost:8000>. Vite chuyển tiếp `/api` tới backend. Chế độ `MOCK=1` tạo điểm số xác định để thử luồng giao diện, luôn có banner đỏ và **không phải kết quả mô hình**.

## Chạy với Qwen và checkpoint

Đặt các checkpoint vào `models/` (thư mục đã được giữ khỏi Git), rồi cài thêm phụ thuộc mô hình bằng `pip install -r backend/requirements-models.txt`:

- `titans_vi_memory.pt` hoặc `titans_v2.pt`
- `ttt_linear_vi_memory.pt` hoặc `ttt_linear_v2.pt`
- tùy chọn `ttt_mlp_v2.pt`

Sau khi cache backbone Qwen, có thể chạy offline:

```sh
MOCK=0 make dev
```

Thiết lập `DEVICE=cuda` hoặc `DEVICE=cpu`, `MODELS_DIR`, `BACKBONE` và `OFFLINE=1` nếu cần. Lần khởi tạo đầu có thể tải Qwen từ Hugging Face. API sẽ báo lỗi rõ nếu thiếu gói phụ thuộc, backbone hoặc checkpoint.

## Build và kiểm tra

```sh
make build
make test
```

`make build` tạo frontend tĩnh ở `backend/static/`; khởi động lại backend để phục vụ bản build. `make test` chạy các kiểm thử tham chiếu và backend khi đã cài dependencies và có cấu hình checkpoint phù hợp.

## Tình trạng và giới hạn hiện tại

- Màn hình kiểm tra phiên, ingest có SSE, xếp hạng ứng viên, báo cáo JSON, các trang tổng quan/Kaggle/giới hạn và benchmark cơ bản đã được dựng.
- `MOCK=1` dùng bộ chấm giả lập xác định, không khởi tạo Qwen2 ngẫu nhiên nhỏ như yêu cầu mục tiêu; kết quả chỉ dùng thử giao diện/API.
- Backend benchmark tính accuracy, MRR, logP và bootstrap CI cho dữ liệu người dùng nhập. Cần bộ test held-out phù hợp để có kết luận; retrieval gate mặc định vẫn đóng.
- Titans v2 Phase 0 đã được nạp vào `models/titans_v2.pt`; smoke test CPU với Qwen2.5-0.5B và API thật đã chạy được. Kết quả fact-mix Kaggle nằm trong `docs/eval_v2.json`; retrieval gate vẫn đóng vì chưa có benchmark paired với window theo tiêu chí SPEC §8.
- Xuất báo cáo phiên JSON và benchmark CSV đã có. Xuất PNG, sinh tự do, lưu bản JSON phiên lên đĩa, và Playwright smoke chưa hoàn thiện.

Xem `docs/SPEC.md` làm nguồn yêu cầu và `docs/DECISIONS.md` để biết các quyết định/giới hạn triển khai.
