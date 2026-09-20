# Jev + LangChain TypeSafe

Một project Python nhỏ dùng Jev làm bộ định tuyến ticket hỗ trợ khách hàng.
`TypeSafeClassifier` hỏi đồng thời ba câu trong **một request**:

- `Choice`: ticket thuộc team nào;
- `Noul`: xác suất ticket khẩn cấp;
- `Score`: mức độ bức xúc của khách hàng.

Sau đó code Python áp dụng policy rõ ràng: confidence thấp thì review thủ
công, ticket khẩn cấp thì đưa vào priority queue, còn lại tự động route.

Jev là decision model, không phải chat model. Vì vậy project dùng nó như một
LangChain `Runnable` thay vì bọc nó thành LLM.

## Chạy project

Yêu cầu Python 3.10 trở lên và một API key từ TypeSafe.

TypeSafe chưa cấp API key vẫn có thể chạy toàn bộ luồng CLI ở chế độ offline:

```bash
uv sync --extra dev
uv run jev-route --mock "URGENT: production API is down!"
```

Mock chỉ là heuristic cố định để kiểm tra integration và policy, không mô phỏng chất
lượng dự đoán của Jev.

Khi đã có API key, chạy Jev thật:

```bash
cp .env.example .env
set -a; source .env; set +a
uv run jev-route "Stripe không kết nối được 3 ngày rồi, production đang dừng!"
```

Kết quả là JSON ổn định để service khác sử dụng:

```json
{
  "department": "technical",
  "department_confidence": 0.94,
  "urgent_probability": 0.91,
  "sentiment_score": 2.3,
  "action": "priority_queue"
}
```

Điều chỉnh policy bằng CLI:

```bash
uv run jev-route --min-confidence 0.8 --urgent-threshold 0.9 "Nội dung ticket"
```

## Test

Test không gọi API thật và không tốn credit:

```bash
uv run pytest
```

`langchain-typesafe` hiện là pre-release, nên dependency được pin tại
`0.0.1a2` để tránh thay đổi API ngoài ý muốn.
