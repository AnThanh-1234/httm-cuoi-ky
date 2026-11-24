Bạn là kỹ sư full-stack. Hãy xây web app **"Kiểm tra truy xuất của memory test-time: Titans vs TTT"** trong repo này.

ĐỌC TRƯỚC (theo thứ tự): `AGENTS.md`, `docs/SPEC.md`, `docs/kaggle_results.json`, `reference/memlm.py`, `reference/tests/test_core_invariants.py`, `docs/PHASE0_retrain.md`.

BỐI CẢNH
Backbone Qwen2.5-0.5B đóng băng + một memory adapter (kiểu Titans hoặc kiểu TTT) cập nhật trọng số nhanh lúc test. Giảng viên nạp tài liệu, đặt câu hỏi, đưa đáp án đúng, và xem mô hình nào xếp đáp án đúng cao nhất. Lõi suy luận đã có sẵn và đã có test ở `reference/` — dùng lại, KHÔNG viết lại.

SỰ THẬT QUAN TRỌNG (phải thể hiện trong UI, không được che)
Ở log Kaggle: TTT-Linear giảm loss Wikipedia tốt hơn Titans (+0.0632 vs +0.0195), nhưng cả hai chưa truy xuất được sự kiện bịa (0/12, như Qwen thuần), còn full-context đạt 12/12. Do đó app là công cụ KIỂM CHỨNG trung thực, không phải demo "khoe". Hiện banner cảnh báo cho tới khi backend tự tính `passed_retrieval_gate=true` (SPEC §8).

CẦN LÀM
1. Backend FastAPI theo SPEC §2–§4: import `reference/memlm.py`; API đủ như bảng; khóa một job GPU; SSE tiến độ; `MOCK=1` dùng Qwen2 ngẫu nhiên nhỏ (UI có banner đỏ).
2. Frontend Vite+React+TS, tiếng Việt, chạy offline, 6 trang theo SPEC §5 (trang "Kiểm tra truy xuất" là chính: dán tài liệu → câu hỏi + 2–6 ứng viên + đánh dấu đáp án đúng → bảng hạng 4 mode, chèn nhiễu theo khoảng cách). Memory Lab vẽ trọng số thay đổi theo token (Titans vs TTT) + heatmap lợi ích theo token.
3. Benchmark hàng loạt có seed, bootstrap CI 95%, xuất CSV/JSON; xuất báo cáo phiên.
4. Test: `pytest` backend (kể cả các bất biến SPEC §9.3–9.4) và 1 Playwright smoke (SPEC §9.8). Thêm test grep đảm bảo không có số kết quả hard-code trong frontend.
5. `Makefile` (`dev`, `build`, `test`), `README.md` (cài đặt, đặt checkpoint vào `models/`, chạy), `docs/DECISIONS.md` (mọi giả định bạn đã chọn).

RÀNG BUỘC
- Không đổi thuật toán trong `reference/`. Không claim đây là Titans/TTT end-to-end; gọi là "memory adapter trên Qwen đóng băng".
- Truy vấn không được sửa trạng thái phiên; mọi phép chấm xác định (không sampling).
- Không thêm tính năng ngoài SPEC §10.

GIAO NỘP
Chạy được bằng `make dev` với `MOCK=1` (không cần GPU/Internet) và với checkpoint thật + Qwen. Cuối cùng liệt kê: test đã chạy và kết quả, phần nào chưa kiểm chứng với Qwen thật, và các quyết định đã tự đưa ra.
