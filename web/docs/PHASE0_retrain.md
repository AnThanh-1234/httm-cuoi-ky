# Phase 0 — Train lại (v2) để memory thực sự truy xuất được
Chạy bởi người dùng trên Kaggle (Claude sẽ tạo notebook v2 nếu được yêu cầu). Web app chỉ cần đọc checkpoint cùng định dạng (`cfg` có thể có thêm `titans_kwargs`).

## Vì sao
Log v1: cả hai memory giảm loss Wikipedia nhưng recall sự kiện bịa = 0/12. Next-token loss trên Wikipedia hầu như không thưởng việc nhớ chính xác một con số vừa đọc, nên adapter không có động lực học việc đó.

## Thay đổi
1. **Mục tiêu train trộn sự kiện (fact-mix)**: mỗi tài liệu chèn 1–3 câu sự kiện bịa (template tiếng Việt, tên/số ngẫu nhiên) ở segment k; ở segment sau ≥ 1 segment hỏi lại bằng prompt; loss trên token đáp án nhân trọng số (vd 5–10×). Dùng **cho cả hai mô hình** để công bằng. Giữ phần loss ngôn ngữ thường.
2. **Tập tách biệt**: template/tên dùng khi train ≠ khi validate/test (không rò rỉ). Tập validation và test cố định, lưu kèm checkpoint.
3. **Titans init** (kiểm soát nhiễu cấu hình): `init_decay_bias≈-5` (gate quên ≈ 0.007, không phải 0.51), `init_momentum_bias≈+2`, `init_adaptive_step_bias≈0`, `default_step_transform_max_lr=1e-2`, `qk_rmsnorm=True`, `max_grad_norm=1.0`. Ghi log giá trị gate α/η/θ sau train (webapp sẽ hiển thị).
4. **Ablation** để tách nguyên nhân: Titans {gate mặc định, gate init mới} × {có/không momentum}; TTT {Linear, MLP}. Mỗi cấu hình ≥ 2 seed.
5. **Đánh giá**: top-1/MRR trong tập 4 ứng viên + logP, khoảng cách 0/1/2/4/7 segment, CI bootstrap; báo cả loss Wikipedia để chắc chắn không hỏng khả năng ngôn ngữ.
6. **Tiêu chí dừng/kết luận**: chỉ kết luận "A tốt hơn B" khi CI 95% của hiệu không chứa 0 trên ≥ 2 seed. Nếu không có mô hình nào vượt `window` thì báo đúng như vậy (kết quả âm vẫn hợp lệ để demo).

## Đầu ra
`titans_v2.pt`, `ttt_linear_v2.pt` (+ `ttt_mlp_v2.pt` nếu có), `eval_v2.json` (cùng schema `kaggle_results.json`), log.
