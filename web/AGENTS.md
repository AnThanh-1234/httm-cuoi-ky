# AGENTS.md — quy tắc cho agent lập trình
- Đọc `docs/SPEC.md` trước; đó là nguồn sự thật. Mâu thuẫn → hỏi/ghi vào `docs/DECISIONS.md`, không tự đoán.
- **Không viết lại** `reference/memlm.py` và `reference/ttt_core.py`; copy/import. Muốn sửa phải kèm test và ghi lý do.
- Không hard-code số kết quả vào frontend/backend; chỉ đọc từ lần chạy thật hoặc `docs/kaggle_results.json`.
- Không thêm tính năng ngoài SPEC §10. Không gọi mạng lúc chạy (trừ tải Qwen lần đầu). Không dùng CDN.
- Mọi thay đổi chạy: `pytest reference/tests tests -q` (cần `DEBUG_CKPT_DIR` hoặc `MOCK=1`) và build frontend phải xanh.
- Không commit checkpoint, token, `.env`. UI tiếng Việt; code/comment tiếng Anh hoặc Việt đều được, nhất quán.
- Khi không chắc về toán/ML: giữ nguyên hành vi của `reference/` và nêu câu hỏi, đừng "tối ưu" tại chỗ.
