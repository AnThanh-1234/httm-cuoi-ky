# models/
Đặt checkpoint từ Kaggle vào đây (không commit lên git):
- titans_vi_memory.pt
- ttt_linear_vi_memory.pt
- (v2 sau Phase 0) titans_v2.pt, ttt_linear_v2.pt
- data_vi.pt (tập eval, tuỳ chọn)
Backbone Qwen/Qwen2.5-0.5B tải từ HuggingFace trên máy có Internet (cache ở ~/.cache/huggingface).
Chạy không cần checkpoint: biến môi trường MOCK=1 dùng Qwen2 ngẫu nhiên nhỏ (chỉ để kiểm thử giao diện/API, UI phải ghi rõ "MOCK").
