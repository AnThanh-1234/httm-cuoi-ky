# Gợi ý xây dựng WebApp demo kiến trúc Titans

## 1. Mục tiêu đồ án

Xây dựng **một WebApp tự phát triển**, cho phép giảng viên trực tiếp nhập dữ liệu, chạy thử mô hình và quan sát các cơ chế được Titans đề cập trong nghiên cứu *Titans: Learning to Memorize at Test Time*.

Demo cần tách biệt:
- **Sản phẩm:** ứng dụng tương tác, không chỉ là notebook hoặc dashboard biểu đồ tĩnh.
- **Cơ chế:** Neural Long-Term Memory, test-time memorization, surprise, momentum, forgetting, phối hợp attention và memory.
- **Bằng chứng:** benchmark trên dữ liệu có đáp án chuẩn, so sánh với baseline công bằng.

> **Lưu ý:** Một module Neural Memory tự viết hoặc một LLM gắn thêm vector database không tự động là mô hình Titans hoàn chỉnh. Chỉ gọi là “Titans” khi triển khai đúng biến thể và quy trình phù hợp; nếu không, dùng tên “Titans-inspired prototype”.

## 2. Các ý tưởng ứng dụng

### Phương án A — Meeting Memory Assistant (ưu tiên)

**Bài toán:** Người dùng nạp nhiều biên bản họp theo thời gian; ứng dụng ghi nhớ các quyết định và cho phép hỏi lại sau nhiều phiên.

**Tính năng:**
1. Tạo dự án và upload transcript TXT/JSON/PDF.
2. Xem timeline các cuộc họp, người phát biểu và sự kiện.
3. Nạp dữ liệu theo từng chunk để quan sát trạng thái bộ nhớ.
4. Hỏi về một quyết định cũ; hiển thị bằng chứng và độ chính xác.
5. Chạy cùng câu hỏi qua baseline và Titans/Titans-inspired model.
6. Mở Memory Inspector để xem surprise, loss, độ lớn cập nhật và hiệu ứng quên.

**Kịch bản:** Cuộc họp đầu tiên quyết định dùng MySQL; cuộc họp sau chuyển sang PostgreSQL vì JSONB; sau nhiều cuộc họp gây nhiễu, hỏi lý do chuyển đổi. Dữ liệu phải được thiết kế để có đáp án chuẩn.

### Phương án B — Project Decision Tracker

**Bài toán:** Theo dõi các quyết định kỹ thuật và thay đổi yêu cầu trong dự án phần mềm.

**Demo:** Nhập lịch sử issue/commit/decision; kiểm tra model có nhớ quyết định cũ, phát hiện thay đổi và truy xuất nguyên nhân khi khoảng cách ngữ cảnh tăng lên hay không.

**Ưu điểm:** Gần thực tế phát triển phần mềm; dễ tạo bộ câu hỏi đánh giá.

### Phương án C — Personal Knowledge Assistant

**Bài toán:** Ghi nhớ ghi chú học tập/công việc qua nhiều phiên.

**Demo:** Nạp từng nhóm ghi chú, đưa vào dữ liệu nhiễu, sau đó hỏi thông tin ở phiên đầu. Cho người dùng quan sát bộ nhớ cập nhật khi thông tin mới xuất hiện.

**Hạn chế:** Nếu chỉ dùng RAG/embedding search thì chưa chứng minh được lợi ích kiến trúc Titans.

### Phương án D — Neural Memory Playground (ứng dụng nghiên cứu tương tác)

**Bài toán:** Cho người dùng tự tạo cặp key–value, phát luồng dữ liệu và đặt truy vấn truy xuất.

**Demo:** Nhập `project_A -> PostgreSQL`, thêm nhiều cặp gây nhiễu, hỏi lại `project_A`; thay đổi độ dài chuỗi, surprise, momentum và forgetting; xem thay đổi trong tỷ lệ recall.

**Ưu điểm:** Khả thi nhất trên Colab T4, dễ kiểm soát dữ liệu và giải thích kết quả. Có thể kết hợp làm một màn hình của Meeting Memory Assistant.

## 3. Những nội dung của Titans cần thể hiện

| Nội dung | Cách người dùng tương tác | Quan sát/đo lường |
|---|---|---|
| Neural Long-Term Memory | Nạp các cặp thông tin theo luồng | Trạng thái memory, khả năng truy xuất key–value |
| Test-time memorization | Thêm thông tin mới khi đang chạy, không cập nhật toàn bộ backbone | Kết quả trước/sau khi cập nhật memory |
| Surprise | Thêm thông tin quen thuộc và thông tin khó dự đoán | Gradient norm / surprise và mức cập nhật |
| Momentum | Bật/tắt hoặc thay đổi hệ số | Độ lớn và độ ổn định cập nhật qua các bước |
| Forgetting | Thay đổi hệ số quên | Recall của thông tin cũ và mới theo thời gian |
| Short-term attention + long-term memory | Tăng số chunk, giới hạn context hiện tại | Recall theo khoảng cách từ dữ kiện đến truy vấn |
| Persistent memory | Bật/tắt các token/tham số persistent đã được học (nếu triển khai) | Ablation có kiểm soát |
| MAC/MAG/MAL | Chọn biến thể được triển khai thật | So sánh accuracy/latency với cùng protocol |

**Quan trọng:** Surprise là tín hiệu dựa trên gradient/sai số của memory, không phải điểm “độ bất ngờ về mặt ngữ nghĩa” được suy ra trực tiếp từ từ khóa. Persistent memory không đồng nghĩa với lịch sử chat được lưu trong database.

## 4. Thiết kế WebApp đề xuất

### Trang 1 — Workspace
- Tạo phiên thử nghiệm, upload tài liệu hoặc sinh dataset mẫu.
- Chọn baseline / Titans-inspired / biến thể Titans đã triển khai.
- Nạp từng chunk và đặt truy vấn.
- Hiển thị kết quả, thông tin nguồn và trạng thái đúng/sai.

### Trang 2 — Memory Inspector
- Biểu đồ surprise theo bước.
- Associative loss, `||ΔW||`, momentum, forgetting.
- Snapshot hoặc khoảng cách giữa các trạng thái memory.
- Cho phép xem trước/sau khi nạp một dữ kiện.

### Trang 3 — Long-Context Playground
- Chèn một fact tại đầu/giữa/cuối chuỗi.
- Chọn số chunk gây nhiễu và độ dài context.
- Đặt câu hỏi truy xuất fact sau khi chạy stream.
- So sánh độ chính xác theo khoảng cách.

### Trang 4 — Benchmark & Ablation
- Cùng dataset, cùng câu hỏi, cùng điều kiện truy cập dữ liệu.
- So sánh accuracy/recall@k, latency, GPU memory, memory state size.
- Bật/tắt momentum, forgetting, persistent memory nếu có triển khai.
- Xuất CSV/JSON kết quả và seed tái lập.

## 5. Model và lựa chọn triển khai

### Lựa chọn 1 — Titans-inspired Neural Memory (dễ thực hiện)
- Giữ SentenceTransformer pretrained để tạo embedding.
- Xây dựng module memory và cơ chế cập nhật có kiểm chứng.
- Thiết kế benchmark associative recall có ground truth.
- **Không gọi đây là Titans Language Model hoàn chỉnh.**

### Lựa chọn 2 — Titans architecture từ thư viện cộng đồng
- Tham khảo `lucidrains/titans-pytorch`: https://github.com/lucidrains/titans-pytorch
- Tham khảo `pafos-ai/titans-trainer`: https://github.com/pafos-ai/titans-trainer
- Thư viện cung cấp mã kiến trúc/huấn luyện, **không đồng nghĩa có checkpoint LLM pretrained đa dụng**.
- Nếu dùng model khởi tạo ngẫu nhiên, cần train trên tác vụ nhỏ để kết quả có ý nghĩa.

### Lựa chọn 3 — Pretrained LLM + memory bên ngoài
- Có thể dùng để xây giao diện hỏi đáp tự nhiên.
- Chỉ nên xem là **baseline sản phẩm hoặc hệ thống lai**, không được trình bày là Titans nguyên bản.

**Khuyến nghị thực tế:** Bắt đầu từ phương án A + D: Meeting Memory Assistant tích hợp Neural Memory Playground. Chạy mô hình nghiên cứu nhỏ trên T4, dùng giao diện web để điều khiển thí nghiệm. Nếu chưa có checkpoint Titans phù hợp, trung thực ghi “Titans-inspired” và giới hạn kết luận.

## 6. Tận dụng notebook `titans_colab_demo-v6.ipynb`

Notebook đã có:
- Nạp transcript COVID từ `covid_1.txt` (có dữ liệu giả lập fallback).
- Tạo embedding bằng `paraphrase-multilingual-MiniLM-L12-v2`.
- Module `TitansNeuralMemoryDetailed` với `W_k`, `W_v`, `W_q`, memory matrix, surprise, momentum, forgetting.
- Biểu đồ và thử nghiệm recall.
- Các baseline Linear Attention, Delta Rule, Sliding Window, Full Cosine.

**Những điểm phải sửa trước khi công bố benchmark:**
1. `W_k`, `W_v`, `W_q` đang khởi tạo ngẫu nhiên, chưa có quy trình học thích hợp.
2. So sánh cosine của một truy vấn với một phát biểu chưa phải benchmark retrieval có đáp án chuẩn.
3. `FullCosineBaseline` hiện có thể tìm chính vector đang xét; cần loại self-match trong phép thử retrieval phù hợp.
4. Titans đang dùng `W_final` để đánh giá toàn bộ các thời điểm; phải tránh future leakage trong online evaluation.
5. Không diễn giải surprise cao là do một từ khóa cụ thể nếu chưa có phép thử nhân quả hoặc phân tích hỗ trợ.
6. Các baseline phải cùng query/target, cùng thời điểm truy cập thông tin và cùng protocol.

## 7. Kịch bản bảo vệ trước giảng viên (5–7 phút)

1. **Giới thiệu:** Titans giải quyết hạn chế của attention window bằng Neural Long-Term Memory.
2. **Nhập dữ liệu:** Tạo một project, upload nhiều cuộc họp hoặc dùng dataset chuẩn.
3. **Stream:** Cho dữ liệu chạy từng chunk; quan sát surprise, loss và memory update.
4. **Truy xuất:** Hỏi một thông tin từ chunk đầu sau nhiều chunk nhiễu.
5. **So sánh:** Chạy cùng truy vấn trên baseline và mô hình có Neural Memory; hiển thị kết quả thực, không giả định Titans thắng.
6. **Ablation:** Tắt momentum/forgetting, chạy lại với cùng seed và giải thích thay đổi.
7. **Kết luận:** Nêu kết quả đo được, giới hạn mô hình và khác biệt giữa prototype với Titans đầy đủ.

## 8. Stack kỹ thuật và lộ trình

**Frontend:** React + TypeScript + Tailwind CSS + Recharts.

**Backend:** Python + FastAPI + PyTorch + SentenceTransformers.

**Lưu trữ:** SQLite cho MVP hoặc PostgreSQL nếu cần nhiều phiên; checkpoint và telemetry lưu file.

**Compute:** Colab T4 cho thử nghiệm/huấn luyện nhỏ; local cho frontend và API. Colab có thể không phù hợp để làm backend online ổn định lâu dài.

**Lộ trình:**
- **Tuần 1:** Sửa protocol đánh giá trong notebook; xây dataset associative recall có ground truth.
- **Tuần 2:** Chuẩn hóa model API và telemetry; thêm baseline công bằng.
- **Tuần 3:** Xây frontend Workspace, Memory Inspector và Long-Context Playground.
- **Tuần 4:** Tích hợp benchmark/ablation, kiểm thử, lưu kết quả và chuẩn bị demo.

## 9. Tiêu chí nghiệm thu

- Người dùng thực sự thao tác trên WebApp tự build.
- Dữ liệu nhập vào làm thay đổi trạng thái Neural Memory quan sát được.
- Có truy vấn truy xuất thông tin xa với ground truth.
- Có ít nhất một baseline được so sánh công bằng.
- Có biểu đồ surprise, loss, memory update và forgetting.
- Có kết quả đo thực tế, có thể chạy lại.
- Có mô tả trung thực: implementation đầy đủ, một biến thể Titans, hay Titans-inspired prototype.

## 10. Tài liệu tham khảo

- Paper Titans: https://arxiv.org/abs/2501.00663
- Google Research – Titans + MIRAS: https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/
- PyTorch implementation cộng đồng: https://github.com/lucidrains/titans-pytorch
- Titans trainer cộng đồng: https://github.com/pafos-ai/titans-trainer

---

**Đề xuất chốt:** Xây **Meeting Memory Assistant + Neural Memory Lab**. Đây là ứng dụng có chức năng thực tế và vẫn cho giảng viên trực tiếp kiểm tra các cơ chế nghiên cứu của Titans. Ưu tiên một biến thể triển khai đúng và benchmark hợp lệ hơn là tuyên bố hỗ trợ toàn bộ MAC/MAG/MAL nhưng chỉ mô phỏng giao diện.
