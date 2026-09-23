# Jev + LangChain TypeSafe

Hai bài toán quyết định chạy trên cùng một mô hình Jev:

- `jev_router` — định tuyến ticket hỗ trợ khách hàng (đang chạy được với API thật);
- `snake_jev` — game rắn săn mồi dựng sẵn để Jev tự chơi (phần nối Jev làm sau).

## Định tuyến ticket

Dùng Jev làm bộ định tuyến ticket hỗ trợ khách hàng.
`TypeSafeClassifier` hỏi đồng thời ba câu trong **một request**:

- `Choice`: ticket thuộc team nào;
- `Noul`: xác suất ticket khẩn cấp;
- `Score`: mức độ bức xúc của khách hàng.

Sau đó code Python áp dụng policy rõ ràng: confidence thấp thì review thủ
công, ticket khẩn cấp thì đưa vào priority queue, còn lại tự động route.

Jev là decision model, không phải chat model. Vì vậy project dùng nó như một
LangChain `Runnable` thay vì bọc nó thành LLM.

## Chạy project

Yêu cầu Python 3.10 trở lên. Key nào cần cho việc gì:

| Biến | Dùng cho |
|---|---|
| `TYPESAFE_API_KEY` | `jev-route`, và `snake --agent jev` (Jev thật) |
| `OPENROUTER_API_KEY` | `snake --agent openrouter` (chat model, không phải Jev) |

Hai key này không thay thế cho nhau. Key OpenRouter (`sk-or-v1-…`) đặt vào
`TYPESAFE_API_KEY` sẽ cho 401, vì nó được gửi tới `api.typesafe.ai`.

TypeSafe chưa cấp API key vẫn có thể chạy toàn bộ luồng CLI ở chế độ offline:

```bash
uv sync --extra dev
uv run jev-route --mock "URGENT: production API is down!"
```

Thêm `--extra ui` nếu muốn chơi game rắn trong cửa sổ.

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

## Game rắn săn mồi

`snake_jev` là môi trường ra quyết định. Mỗi tick nó biến bàn cờ thành text,
hỏi một câu `Choice` gồm bốn hướng, rồi để code Python quyết định có nghe theo
hay không — đúng cách `jev_router` xử lý ticket.

Luật nằm hết trong `game.py`. Cửa sổ pygame, bản terminal và các agent chỉ là ba
cách nhìn vào cùng một engine, nên thứ bạn xem trên màn hình đúng bằng thứ agent
nhìn thấy. Cửa sổ cần extra `ui`; mọi thứ còn lại chạy không cần pygame.

Chơi trong cửa sổ:

```bash
uv sync --extra ui
uv run snake ui
```

Mũi tên hoặc WASD để lái, `space` tạm dừng, `r` chơi lại, `q` thoát. Chỉnh bàn
cờ và tốc độ bằng `--width/--height`, `--tick` (giây mỗi bước) và `--cell` (số
pixel một ô).

Xem agent chơi trong cùng cửa sổ đó — đây là cách sẽ dùng để theo dõi Jev:

```bash
uv run snake ui --agent greedy --tick 0.05
```

Bản terminal vẫn còn, tiện khi chỉ có SSH:

```bash
uv run snake play --width 16 --height 16
```

Đo điểm trên nhiều ván, không cần cửa sổ:

```bash
uv run snake auto --agent greedy --episodes 20 --seed 0
uv run snake auto --agent greedy --watch          # xem ngay trong terminal
```

`--seed` cố định chuỗi thức ăn, nên hai agent chạy cùng seed là so sánh được với
nhau. `greedy` (BFS tới mồi, kẹt thì chọn hướng còn nhiều ô trống nhất) là mốc
để đánh giá Jev; `random` là sàn.

### Jev nhìn thấy gì

Xem đúng chuỗi text sẽ gửi cho Jev:

```bash
uv run snake show --seed 2
```

```
Board 12x12, column 0 is the left edge and row 0 the top.
+------------+
|............|
|..*.........|
|............|
|............|
|............|
|............|
|....oo@.....|
|............|
|............|
|............|
|............|
|............|
+------------+
Head at (6, 6), moving RIGHT.
Snake length 3, score 0, step 0.
Food at (2, 1): 5 up and 4 left.
Move options:
- UP: safe
- DOWN: safe
- LEFT: not allowed, that would reverse into the neck
- RIGHT: safe
```

Nước đi chết được tính sẵn bằng Python và ghi thẳng vào prompt. Va chạm là phép
kiểm tra chính xác và rẻ, nên chỉ hỏi Jev phần thật sự cần phán đoán: đi hướng
nào thì tốt.

### Cho Jev chơi

```bash
uv run snake ui --agent jev --mock          # offline, không tốn credit
uv run snake ui --agent jev                 # cần TYPESAFE_API_KEY
```

Cửa sổ mở thêm một panel bên phải ghi lại từng câu trả lời của Jev:

- nước vừa chọn và confidence;
- phân phối xác suất trên cả bốn hướng, hướng chết đánh dấu `x` màu đỏ;
- bộ đếm calls / fallbacks / errors, latency trung bình, token đã dùng;
- danh sách các lượt gần nhất, lượt nào bị từ chối thì hiện màu vàng kèm lý do.

Jev được gọi trên thread riêng nên cửa sổ không đứng trong lúc chờ API; panel
hiện `thinking…` và bàn cờ chỉ nhích khi câu trả lời về.

Cùng bộ cờ đó chạy không cần cửa sổ, in ra JSON gồm toàn bộ decision log:

```bash
uv run snake auto --agent jev --mock --width 8 --height 8 --seed 3
```

`--min-confidence` đặt ngưỡng bỏ qua câu trả lời lưỡng lự, `--timeout` giới hạn
thời gian chờ mỗi request (mặc định 10s, ngắn hơn mức 30s mặc định của thư viện
để một request treo không làm đứng bàn cờ).

`--mock` dùng `MockMoveClassifier`: một heuristic cố định trả về đúng shape của
`TypeSafeClassifier`, để kiểm tra phần nối dây, policy và panel mà không cần
key. Nó đọc chuỗi prompt chứ không đọc `GameState`, nên nếu prompt thiếu thông
tin thì mock cũng sai theo — đúng như Jev sẽ sai.

### Chạy bằng LLM qua OpenRouter (không phải Jev)

```bash
uv run snake ui --agent openrouter
uv run snake auto --agent openrouter --width 6 --height 6 --model qwen/qwen3-30b-a3b-instruct-2507
```

Cần `OPENROUTER_API_KEY`. Mặc định là `openai/gpt-4o-mini`; đổi bằng `--model`,
model nào cũng được miễn hỗ trợ `top_logprobs`.

Đây **không phải Jev**. OpenRouter không phục vụ Jev — nó định tuyến tới các
chat model. Điểm khác quan trọng nhất là chỗ lấy xác suất: Jev trả thẳng một
phân phối có hiệu chuẩn trên các nhãn m đưa, còn ở đây phân phối được đọc từ
`top_logprobs` của token đầu tiên mà model sinh ra. Nhiều token viết ra cùng
một nước (`UP`, ` UP`, `Up`) được cộng gộp rồi chuẩn hóa lại trên bốn nhãn.

Cách này không hỏi model tự chấm điểm chắc chắn — con số nó tự khai là số bịa.
Nhưng logprobs của chat model cũng không được hiệu chuẩn như Jev, nên hãy đọc
cột confidence ở đây như một chỉ dấu, không phải một phép đo.

Nếu model trả lời dài dòng thay vì một từ, phân phối token vẫn cho biết nó
nghiêng về hướng nào, và nước đó được dùng. Chỉ khi không có cả hai thì lượt đó
mới rơi về agent dự phòng.

Chi phí và tốc độ thực đo với `gpt-4o-mini`: khoảng 196 token input và
$0.00003 mỗi nước, độ trễ ~2 giây. Tức là một ván 6x6 mất vài phút và tốn vài
cent; board 12x12 thì lâu hơn nhiều. Panel hiện tổng chi phí khi API có báo về.

### Policy: Jev phán đoán, code quyết định

`JevAgent` nhận bất cứ object nào có `.invoke(text)`, nên game không phụ thuộc
trực tiếp vào `langchain-typesafe`, và cùng một policy dùng được cho cả Jev lẫn
OpenRouter:

```python
from langchain_typesafe import TypeSafeClassifier
from snake_jev import JevAgent, SnakeGame, build_move_question, run_episode

agent = JevAgent(
    TypeSafeClassifier(questions={"move": build_move_question()}),
    min_confidence=0.6,
)
print(run_episode(SnakeGame(12, 12, seed=0), agent).to_dict())
print(agent.stats)          # calls, fallbacks, errors, latency, tokens
print(agent.log[-1])        # JevDecision gần nhất
```

Câu trả lời bị thay bằng agent dự phòng trong bốn trường hợp, mỗi trường hợp
ghi một `note` riêng vào log:

| note | nghĩa |
|---|---|
| `unsafe move` | hướng đó chết ngay lượt sau |
| `low confidence` | dưới `--min-confidence` |
| `unparsable move` / `malformed response` | trả về thứ không phải một trong bốn hướng |
| tên exception | API lỗi, timeout, mất mạng |

Lỗi mạng chỉ tốn một nước đi chứ không làm hỏng ván. Chạy với key sai sẽ thấy
cả log là `TypeSafeAuthenticationError` và greedy gánh toàn bộ ván — đó là cách
nhanh nhất để biết key có vấn đề.

## Test

Test không gọi API thật và không tốn credit:

```bash
uv run pytest
```

`langchain-typesafe` hiện là pre-release, nên dependency được pin tại
`0.0.1a2` để tránh thay đổi API ngoài ý muốn.
