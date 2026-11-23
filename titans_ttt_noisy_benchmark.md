# Kế hoạch thực nghiệm Titans vs TTT-Linear với dữ liệu nhiễu

## Mục tiêu

Thực hiện hai thí nghiệm độc lập trên Kaggle, tận dụng notebook `compare_titans_vs_ttt_kaggle.ipynb`:

1. **Benchmark checkpoint hiện tại:** Không train lại; đo khả năng truy xuất fact khi có distractor.
2. **Train với dữ liệu nhiễu:** Train Titans và TTT-Linear trên cùng dữ liệu Wikipedia + synthetic QA, sau đó đánh giá trên tập fact mới.

## 1. Giai đoạn 1: Benchmark checkpoint hiện tại

Notebook hiện tại có benchmark truy xuất sự kiện theo khoảng cách, chủ yếu dùng Wikipedia làm filler. Cần bổ sung **nhiễu cạnh tranh thông tin**.

### Ví dụ

**Fact mục tiêu:** Theo ghi chép Delta, định luật Zorvan cho biết hợp kim Vexa có hệ số truyền nhiệt 7,31.

**Distractor:**

- Định luật Kelmar: hợp kim Vexa có hệ số 8,42.
- Định luật Zorvan: hợp kim Lorin có hệ số 5,21.
- Định luật Tavris: hợp kim Vexa có hệ số 3,91.
- Văn bản Wikipedia không liên quan.

**Truy vấn:** Theo định luật Zorvan, hợp kim Vexa có hệ số truyền nhiệt bao nhiêu?

**Đáp án:** 7,31.

### Code benchmark

Thay cell benchmark bắt đầu bằng `NAMES = ["Zorvan"...]` trong notebook. Đoạn code sau giả định notebook đã định nghĩa `NAMES`, `METALS`, `SEG_LEN`, `encode`, `filler_tokens`, `answer_score`, `adapters`, `PRETTY`, `use`, và đã import `torch`, `numpy as np`, `random`.

```python
# ===== NOISY FACT RETRIEVAL BENCHMARK =====
N_TRIALS = 100
DISTANCES = [0, 1, 2, 4, 7]
DISTRACTORS_PER_SEG = 4
SEED_NOISE = 2026

def make_record(name, metal, value):
    return (
        f"Theo ghi chép Delta, định luật {name} "
        f"cho biết hợp kim {metal} có hệ số truyền nhiệt "
        f"bằng {value} đơn vị."
    )

def make_prompt(name, metal):
    return (
        f"Câu hỏi: Theo định luật {name}, hợp kim {metal} "
        f"có hệ số truyền nhiệt bao nhiêu? Trả lời:"
    )

def pack_segment(text, rng):
    ids = encode(text + "\n")
    if len(ids) > SEG_LEN:
        ids = ids[:SEG_LEN]
    if len(ids) < SEG_LEN:
        extra = filler_tokens(rng, SEG_LEN - len(ids))
        ids = ids + extra.tolist()
    return torch.tensor(ids[:SEG_LEN], dtype=torch.long)

def make_noisy_trial(rng, distance):
    target_name = rng.choice(NAMES)
    target_metal = rng.choice(METALS)
    target_value = f"{rng.randint(1,9)},{rng.randint(10,99)}"
    target_fact = make_record(target_name, target_metal, target_value)
    segments = [pack_segment(target_fact, rng)]

    pairs = [
        (n, m) for n in NAMES for m in METALS
        if (n, m) != (target_name, target_metal)
    ]
    decoys = rng.sample(
        pairs, min(distance * DISTRACTORS_PER_SEG, len(pairs))
    )
    for s in range(distance):
        records = []
        for name, metal in decoys[
            s * DISTRACTORS_PER_SEG:(s + 1) * DISTRACTORS_PER_SEG
        ]:
            value = f"{rng.randint(1,9)},{rng.randint(10,99)}"
            records.append(make_record(name, metal, value))
        segments.append(pack_segment("\n".join(records), rng))

    context = torch.cat(segments)
    prompt = make_prompt(target_name, target_metal)
    answer = f" {target_value}"
    return context, encode(prompt), encode(answer)

modes = (
    [("window", "Qwen thuần")]
    + [("mem:" + k, PRETTY[k]) for k in adapters]
)
scores = {mode: {d: [] for d in DISTANCES} for mode, _ in modes}
accuracies = {mode: {d: [] for d in DISTANCES} for mode, _ in modes}

for d in DISTANCES:
    for trial in range(N_TRIALS):
        rng = random.Random(SEED_NOISE + d * 10000 + trial)
        ctx, prompt, answer = make_noisy_trial(rng, d)
        for mode, name in modes:
            if mode.startswith("mem:"):
                m = use(mode[4:])
                lp, prob, ok = answer_score(m, ctx, prompt, answer, "memory")
            else:
                m = use("titans")
                lp, prob, ok = answer_score(m, ctx, prompt, answer, "window")
            scores[mode][d].append(lp)
            accuracies[mode][d].append(float(ok))
    print("Hoàn thành distance:", d)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for mode, name in modes:
    mean_lp = [np.mean(scores[mode][d]) for d in DISTANCES]
    acc = [np.mean(accuracies[mode][d]) for d in DISTANCES]
    ax[0].plot(DISTANCES, mean_lp, marker="o", label=name)
    ax[1].plot(DISTANCES, acc, marker="o", label=name)
ax[0].set_title("Log probability of correct answer")
ax[1].set_title("Exact answer accuracy")
ax[0].set_ylabel("Log probability (cao hơn tốt hơn)")
ax[1].set_ylabel("Accuracy")
for a in ax:
    a.set_xlabel("Số segment distractor")
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()

for mode, name in modes:
    print("\n", name)
    for d in DISTANCES:
        print(
            f"Distance {d}: "
            f"logP={np.mean(scores[mode][d]):.3f} | "
            f"accuracy={np.mean(accuracies[mode][d]):.2%}"
        )
```

**Lưu ý kỹ thuật:** Đây là code mẫu cần đối chiếu với API thực tế của notebook trước khi chạy. `answer_score` có thể đang trả về một chỉ số `ok` khác exact-match generation; cần kiểm tra định nghĩa hàm. Đồng thời, `pack_segment` có thể cắt mất distractor nếu quá dài; cần kiểm tra số token và độ phủ nhiễu. Đảm bảo bộ nhớ được reset giữa các trial và không có rò rỉ đáp án từ prompt.

### Diễn giải

| Quan sát | Diễn giải sơ bộ |
|---|---|
| Titans giảm chậm hơn TTT khi tăng distractor | Titans có dấu hiệu chống nhiễu tốt hơn |
| TTT giảm chậm hơn Titans | TTT có lợi thế trong cấu hình này |
| Cả hai giảm mạnh | Cả hai có thể yếu trước interference |
| Cả hai gần 0% | Có thể benchmark quá khó hoặc checkpoint chưa học truy xuất |

Chưa thể kết luận forgetting từ benchmark interference đơn lẻ. Nên đánh giá thêm theo vị trí fact, nhiều seed, hard negatives và khoảng tin cậy 95%.

## 2. Giai đoạn 2: Train với dữ liệu nhiễu

Mỗi tài liệu gồm 8 segment:

- **Segment 0:** Fact mục tiêu (ví dụ Zorvan – Vexa – 7,31).
- **Segment 1–6:** Distractor (fact cạnh tranh) và văn bản Wikipedia.
- **Segment 7:** Query + Answer của fact mục tiêu.

Giữ causal language modeling loss, nhưng cần kiểm tra loss có thực sự tạo tín hiệu học từ đáp án ở segment cuối hay không. Có thể cân nhắc mask/weight phần query-answer để tránh loss trên filler lấn át tín hiệu truy xuất.

### Cấu hình đề xuất

| Thành phần | Hiện tại | Thí nghiệm mới |
|---|---|---|
| Backbone | Qwen2.5-0.5B | Giữ nguyên |
| Memory | Titans / TTT | Giữ nguyên |
| Dữ liệu | Wikipedia | Wikipedia + synthetic QA |
| Số segment | 8 | 8 |
| Segment đầu | Wikipedia | Fact mục tiêu |
| Segment giữa | Wikipedia | Distractor + Wikipedia |
| Segment cuối | Wikipedia | Query + Answer |
| Optimizer | AdamW | Giữ nguyên |
| Số bước | 3.000 | Giữ nguyên ban đầu |

Tỷ lệ khởi đầu đề xuất: **50% Wikipedia thông thường, 50% synthetic QA**. Đây là một giả thuyết cấu hình cần kiểm chứng, không phải tỷ lệ đã được tối ưu.

Tách train/eval theo fact và đáp án, đồng thời tránh rò rỉ template nếu muốn kiểm tra khả năng khái quát hóa. Hai kiến trúc dùng cùng dữ liệu, thứ tự mẫu, seed, token budget và quy trình đánh giá.

## 3. Ma trận checkpoint

| Checkpoint | Dữ liệu train |
|---|---|
| Titans Original | Wikipedia |
| TTT Original | Wikipedia |
| Titans Noisy | Wikipedia + QA nhiễu |
| TTT Noisy | Wikipedia + QA nhiễu |

Đánh giá cả bốn checkpoint trên tập sạch và tập nhiễu chưa thấy khi train. Lưu kết quả từng trial, tính accuracy, log probability và khoảng tin cậy 95%.

## 4. Kiểm soát chất lượng benchmark

- Có ít nhất một distractor **cùng định luật, khác hợp kim**.
- Có ít nhất một distractor **cùng hợp kim, khác định luật**.
- Có thêm fact hoàn toàn không liên quan.
- Tránh distractor vô tình trùng fact mục tiêu hoặc tạo đáp án mâu thuẫn cho cùng cặp khóa.
- Kiểm tra độ dài token của từng segment và tỷ lệ fact thực sự còn trong context.
- Giữ nguyên số lượng và loại nhiễu khi so sánh khoảng cách, nếu muốn tách riêng ảnh hưởng của khoảng cách khỏi mật độ nhiễu.
- Báo cáo kết quả qua nhiều seed và khoảng tin cậy.

## 5. Thứ tự triển khai

1. Chạy benchmark mới trên hai checkpoint hiện tại.
2. Kiểm tra độ suy giảm theo interference.
3. Nếu cần, tạo dataset noisy với train/eval tách biệt.
4. Train lại hai kiến trúc dưới điều kiện tương đương.
5. Chạy benchmark sạch và nhiễu trên bốn checkpoint.
6. Phân tích trade-off giữa truy xuất, chống nhiễu và khả năng khái quát hóa.

**Khuyến nghị:** Ưu tiên giai đoạn 1 để có baseline trước khi tốn thêm tài nguyên GPU cho việc train lại.
