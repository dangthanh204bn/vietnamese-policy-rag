# Vietnamese Policy RAG

Ứng dụng RAG hỏi đáp trên tài liệu TXT và PDF có lớp văn bản, kết hợp tìm kiếm ngữ nghĩa, reranking và mô hình ngôn ngữ chạy qua Ollama. Người dùng nhập câu hỏi trên Streamlit, nhận câu trả lời kèm nhãn trích dẫn và mở nội dung nguồn; với PDF, hộp nguồn hiển thị thêm số trang.

Dự án được xây dựng để học cách triển khai một ứng dụng AI từ xử lý tài liệu đến API, giao diện và đánh giá kết quả. Các bước chunking, tính cosine similarity, chọn kết quả và xây dựng context được viết thành các module Python riêng để dễ kiểm tra luồng xử lý.

## Bài toán và dữ liệu

Tài liệu chính là [store_policy_demo.txt](store_policy_demo.txt), mô tả chính sách giao hàng, đổi sản phẩm, hoàn tiền và bảo hành của một cửa hàng giả lập. Cửa hàng, địa chỉ và thông tin liên hệ trong tài liệu đều là dữ liệu thực hành.

[sample.pdf](sample.pdf) là tài liệu thực hành **13 trang có lớp văn bản**, được biên soạn từ chính sách TXT của dự án. Trang đầu là phần giới thiệu và mục lục; 12 trang sau trình bày từng mục chính sách, kèm diễn giải và tình huống minh họa. Cửa hàng, địa chỉ, email và tình huống đều là giả lập.

PDF này dùng để kiểm tra truy xuất nội dung ở các trang khác nhau, trả lời theo nguồn và giữ số trang qua API đến giao diện. Bốn phản hồi thử thủ công ban đầu nằm trong [pdf_evaluation_result.json](pdf_evaluation_result.json). Bộ câu hỏi tương ứng được khai báo trong [pdf_evaluation_cases.py](pdf_evaluation_cases.py), và kết quả lần chạy đánh giá tự động qua API được lưu trong [pdf_evaluation_report.json](pdf_evaluation_report.json).

Ví dụ câu hỏi:

- Sản phẩm được bảo hành bao lâu?
- Điều kiện để được đổi sản phẩm trong 14 ngày là gì?
- Sản phẩm không bị lỗi nhưng tôi muốn đổi sang mẫu khác thì có được không?
- Phí vận chuyển khi gửi sản phẩm đi bảo hành là bao nhiêu?

Câu hỏi cuối dùng để kiểm tra cách hệ thống phản hồi khi tài liệu thiếu thông tin. Prompt yêu cầu mô hình trả lời theo tài liệu, trích dẫn nguồn và nêu rõ khi không đủ thông tin.

## Chức năng hiện có

- Đọc TXT UTF-8 hoặc trích xuất văn bản từ PDF bằng pypdf, chuẩn hóa đầu ra thành các record có nội dung và metadata nguồn.
- Dùng mỗi đoạn TXT hoặc mỗi trang PDF có văn bản làm parent, sau đó chia thành các child chunk theo token.
- Tạo embedding cho child chunk; lưu metadata vào JSON và vector vào NumPy `.npy`.
- Truy xuất bằng cosine similarity, lọc theo ngưỡng và chọn các ứng viên có điểm cao nhất.
- Loại các ứng viên trùng parent, rerank theo nội dung đầy đủ của parent rồi dùng nội dung đó làm context.
- Sinh câu trả lời tiếng Việt qua Ollama với nhãn nguồn dạng `[chunk_23]`.
- Giữ số trang PDF trong metadata, context, nguồn trả về của API và nhãn hộp nguồn trên Streamlit.
- Cung cấp FastAPI backend, Streamlit UI và chế độ hỏi đáp trên terminal.
- Chọn index TXT hoặc PDF cho API và CLI bằng `INDEX_DIR` trong `config.py`.
- Lưu kết quả đánh giá vào JSON; kiểm tra cụm đáp án mong đợi và ID trích dẫn ngoài context.
- Chạy bộ kiểm tra PDF qua API với nhóm cách diễn đạt được chấp nhận, đối chiếu nguồn được trích theo tên file và số trang, ghi báo cáo JSON cùng lỗi từng ca.

## Luồng xử lý

### Chuẩn bị index

```mermaid
flowchart LR
    A["TXT UTF-8 hoặc PDF có lớp văn bản"] --> B["Nạp văn bản và metadata nguồn"]
    B --> C["Parent: đoạn TXT hoặc trang PDF"]
    C --> D["Chia child chunk theo token"]
    D --> E["Embedding child chunk"]
    D --> F["Metadata JSON"]
    E --> G["Vector NumPy"]
```

Record mới lưu `chunk_id`, `text`, `parent_id`, `parent_text`, `source` và `page_number`. Với TXT, `page_number` là `None`; với PDF, số trang bắt đầu từ 1 và giữ theo vị trí trong tài liệu gốc, kể cả khi bỏ qua trang không có văn bản. Index TXT cũ thiếu trường này vẫn được đọc nhờ cách lấy metadata bằng `.get()`.

| Index mẫu | Nguồn | Số parent | Số child chunk | Kích thước mảng vector |
| --- | --- | --- | --- | --- |
| `index/` | `store_policy_demo.txt` | 14 | 37 | `37 × 384` |
| `index_pdf/` | `sample.pdf` | 13 | 84 | `84 × 384` |

### Trả lời câu hỏi

```mermaid
flowchart TD
    A[Streamlit hoặc CLI] --> B[Embedding câu hỏi]
    B --> C[Cosine với toàn bộ child vector]
    C --> D[Lọc ngưỡng và lấy tối đa 16 child]
    D --> E[Loại trùng parent]
    E --> F[Rerank cặp câu hỏi và parent text]
    F --> G[Lấy tối đa 3 parent làm context]
    G --> H{Có kết quả?}
    H -- Có --> I[Prompt và Ollama]
    I --> J[Câu trả lời và nguồn]
    H -- Không --> K[Thông báo không có đoạn đạt ngưỡng]
```

Chunk nhỏ giúp tìm vị trí liên quan. Parent cung cấp thêm các điều kiện xung quanh để câu trả lời có đủ căn cứ. Reranker chấm cặp `[query, parent_text]`; context cuối cùng cũng chứa `parent_text`.

`chunk_id` trong trích dẫn là ID của child được chọn làm điểm tham chiếu. Vì vậy, nội dung nguồn hiển thị có thể dài hơn child chunk ban đầu. `retrieved_sources` chứa toàn bộ nguồn được đưa vào context, kể cả nguồn không được câu trả lời trích dẫn.

Với PDF, context và nguồn API có thêm số trang. Nhãn `[chunk_ID]` vẫn là ID của child, không phải số trang. Với cấu hình hiện tại, mỗi câu hỏi dùng tối đa 3 parent; khi đọc PDF, đó là tối đa 3 trang được chọn.

## Công nghệ và cấu hình

| Thành phần | Công nghệ / giá trị |
| --- | --- |
| Embedding | [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), chạy qua FastEmbed |
| Reranker | [mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1), dùng CrossEncoder trên CPU |
| Sinh câu trả lời | [Qwen3 4B](https://ollama.com/library/qwen3:4b) qua Ollama `/api/chat`, không streaming |
| Chunking khi build index | 64 token/chunk, overlap 16 token; chưa tính special tokens |
| Truy xuất | `RETRIEVAL_K = 16`, `RETRIEVAL_MIN_SCORE = 0.3` |
| Sau reranking | `RERANK_TOP_N = 3` |
| Đọc PDF | [pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) |
| Backend / UI | FastAPI / Streamlit |
| Lưu index | JSON metadata và NumPy `float32` array |

Các tham số truy xuất và lựa chọn index nằm trong [config.py](config.py). Tên model và tham số chunking hiện vẫn được khai báo trong các script tương ứng.

`RETRIEVAL_K` là số **child chunk trước khi loại trùng parent**, không phải số đoạn parent. Điểm của reranker có thể âm và không cùng thang đo với cosine similarity; ngưỡng `0.3` chỉ dùng ở bước truy xuất ban đầu.

### Chọn index để hỏi đáp

Cấu hình hiện tại trong `config.py` chọn PDF:

```python
INDEX_DIR = "index_pdf"
```

| Giá trị `INDEX_DIR` | Index được API và CLI đọc | `/health` với dữ liệu mẫu |
| --- | --- | --- |
| `"index"` | Chính sách TXT trong `index/` | `chunk_count: 37` |
| `"index_pdf"` | PDF 13 trang trong `index_pdf/` | `chunk_count: 84` |

Đổi giá trị rồi **khởi động lại API hoặc CLI** để nạp index mới. Mỗi lần chạy chọn một index. Trên Streamlit, gửi một câu hỏi mới để nhận phản hồi từ bộ dữ liệu vừa chọn; phản hồi trước đó có thể vẫn còn trong session.

`INDEX_DIR` điều khiển index được đọc khi hỏi đáp. Hai script build bên dưới có nguồn và thư mục lưu riêng; chúng không đổi đích lưu theo biến này.

## Cài đặt trên Windows PowerShell

Môi trường đã chạy dự án: **Windows, Python 3.12.10**. Thực hiện các lệnh trong thư mục chứa README này. Các lệnh dùng trực tiếp Python của `.venv`, nên không cần kích hoạt môi trường bằng `Activate.ps1`.

Để tải code từ GitHub:

```powershell
git clone https://github.com/dangthanh204bn/vietnamese-policy-rag.git
cd vietnamese-policy-rag
```

### 1. Tạo môi trường Python

Nếu đã có `.venv` hoạt động, có thể bỏ qua bước tạo môi trường.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Cài thư viện

Reranker hiện chạy trên CPU. Cài PyTorch CPU, sau đó cài các thư viện ứng dụng:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tham khảo [hướng dẫn cài PyTorch](https://pytorch.org/get-started/locally/) và [các lệnh cài bản CPU](https://pytorch.org/get-started/previous-versions/).

Các thư viện ứng dụng, gồm `pypdf==6.21.4`, được cố định phiên bản trong [requirements.txt](requirements.txt). PyTorch CPU được cài riêng theo lệnh phía trên và hiện chưa cố định phiên bản.

Phiên bản ghi nhận trên môi trường phát triển:

| Thư viện | Phiên bản |
| --- | --- |
| fastembed | 0.9.0 |
| tokenizers | 0.23.3 |
| numpy | 2.5.4 |
| requests | 2.34.2 |
| torch | 2.14.1+cpu |
| sentence-transformers | 6.2.0 |
| fastapi | 0.162.0 |
| pydantic | 2.14.1 |
| uvicorn | 0.54.0 |
| streamlit | 1.66.1 |
| pypdf | 6.21.4 |

Đây là thông tin môi trường đã sử dụng; `requirements.txt` chưa khóa toàn bộ phụ thuộc bắc cầu hoặc phiên bản PyTorch. Kết quả thử cài mới bản TXT được ghi riêng ở phần kiểm tra bên dưới.

### 3. Chuẩn bị Ollama

Cài [Ollama cho Windows](https://docs.ollama.com/windows), mở ứng dụng Ollama và tải model:

```powershell
ollama --version
ollama pull qwen3:4b
ollama list
```

Nếu Ollama chưa chạy nền, có thể khởi động trong một terminal riêng:

```powershell
ollama serve
```

Chỉ cần một Ollama server đang chạy. Endpoint ứng dụng sử dụng là `http://localhost:11434/api/chat` trong [llm_client.py](llm_client.py).

Chạy model local theo cách này không cần API key trả phí. Lần đầu sử dụng vẫn cần kết nối mạng để tải model Ollama, embedding, tokenizer và reranker. Xem thêm [hướng dẫn Ollama](https://docs.ollama.com/quickstart).

### 4. Tạo index TXT

```powershell
.\.venv\Scripts\python.exe -X utf8 build_index.py
```

Script đọc `store_policy_demo.txt`, tạo embedding và lưu:

```text
index/
├── chunk_records.json
└── chunk_vectors.npy
```

Với dữ liệu hiện tại, log sẽ báo 37 chunk và 37 vector. Khi sửa tài liệu, cách chunking hoặc embedding model, cần **build lại index rồi khởi động lại API**. Index hiện chưa tự phát hiện thay đổi của dữ liệu hay model.

### 5. Tạo index PDF

```powershell
.\.venv\Scripts\python.exe -X utf8 pdf_index_demo.py
```

Script đọc `sample.pdf` và lưu:

```text
index_pdf/
├── chunk_records.json
└── chunk_vectors.npy
```

Với PDF mẫu hiện tại, kết quả là **84 chunk và 84 vector**, mỗi vector có 384 chiều. Có 13 parent, tương ứng với 13 trang có văn bản. API và CLI đang chọn index này qua `INDEX_DIR = "index_pdf"` trong `config.py`.

Sau khi tạo lại index, khởi động lại API hoặc CLI để nạp dữ liệu mới. Để hỏi trên TXT, đổi thành `INDEX_DIR = "index"` rồi khởi động lại.

Để thử tài liệu khác, sửa biến `file_path` trong script build tương ứng rồi tạo lại index. PDF đầu vào cần có lớp văn bản để trích xuất; bước này dùng pypdf, chưa thực hiện OCR cho trang chỉ chứa ảnh scan.

## Chạy ứng dụng

Đảm bảo Ollama đang chạy. Mở hai terminal trong thư mục dự án và giữ cả hai tiến trình hoạt động.

**Terminal 1 — FastAPI:**

```powershell
.\.venv\Scripts\python.exe -X utf8 -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Backend tải index, embedding model và reranker một lần khi tiến trình server khởi động. Mỗi câu hỏi vẫn cần tạo embedding cho query, truy xuất, rerank và gọi Ollama nếu có context.

**Terminal 2 — Streamlit:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui.py
```

- Giao diện: [http://localhost:8501](http://localhost:8501).
- Swagger UI để thử API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- Kiểm tra backend: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

Nhập câu hỏi, bấm **Gửi câu hỏi** và mở các hộp nguồn để đối chiếu câu trả lời. UI giữ phản hồi gần nhất trong session; các câu hỏi được xử lý độc lập, chưa có lịch sử hội thoại đưa vào model. Tham khảo [cách chạy Streamlit](https://docs.streamlit.io/get-started/installation/command-line).

Có thể dùng CLI thay cho API và UI:

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py
```

CLI cũng đọc index được chọn bởi `INDEX_DIR`. Phần nguồn ở CLI hiện in tên file, chunk ID và nội dung parent; nhãn số trang riêng được hiển thị trên Streamlit và trong nguồn API.

Nhập `exit` để thoát CLI. Dùng `Ctrl+C` trong terminal tương ứng để dừng API hoặc Streamlit. Chạy các lệnh từ đúng thư mục dự án, nhất là khi có nhiều bản clone trên máy.

## API

### `GET /health`

Với `INDEX_DIR = "index"` và tài liệu TXT mẫu:

```json
{
  "status": "ok",
  "chunk_count": 37
}
```

Với cấu hình hiện tại `INDEX_DIR = "index_pdf"` và PDF 13 trang:

```json
{
  "status": "ok",
  "chunk_count": 84
}
```

Endpoint này xác nhận backend đã tải index; chưa kiểm tra khả năng kết nối hoặc sinh câu trả lời của Ollama.

### `POST /ask`

Ví dụ request khi chọn index TXT (`INDEX_DIR = "index"`):

```json
{
  "query": "Sản phẩm được bảo hành bao lâu?"
}
```

Ví dụ response, đã rút ngắn nội dung và danh sách nguồn để minh họa cấu trúc:

```json
{
  "received_query": "Sản phẩm được bảo hành bao lâu?",
  "answer": "Sản phẩm được bảo hành trong 18 tháng kể từ ngày giao hàng thành công được ghi nhận trên đơn. [chunk_23]",
  "retrieved_sources": [
    {
      "chunk_id": 23,
      "source": "store_policy_demo.txt",
      "text": "8. THỜI HẠN VÀ PHẠM VI BẢO HÀNH\nSản phẩm được bảo hành trong 18 tháng kể từ ngày giao hàng thành công được ghi nhận trên đơn...",
      "page_number": null
    }
  ]
}
```

Response thực tế trả đầy đủ `parent_text` cho từng nguồn được chọn. Câu trả lời có thể thay đổi giữa các lần sinh.

### Ví dụ hỏi trên PDF

Sau khi tạo index PDF và khởi động API với `INDEX_DIR = "index_pdf"`, gửi:

```json
{
  "query": "Khoản hoàn tiền được xử lý trong bao lâu và tính từ mốc nào?"
}
```

Phản hồi đã ghi nhận trong lần chạy thử, rút ngắn nội dung nguồn và chỉ giữ nguồn được trích dẫn để minh họa:

```json
{
  "received_query": "Khoản hoàn tiền được xử lý trong bao lâu và tính từ mốc nào?",
  "answer": "Khoản hoàn tiền được xử lý trong 5 ngày làm việc kể từ ngày phương án hoàn tiền được chấp thuận. [chunk_51]",
  "retrieved_sources": [
    {
      "chunk_id": 51,
      "source": "sample.pdf",
      "text": "...Khoản hoàn tiền được xử lý trong 5 ngày làm việc kể từ ngày phương án hoàn tiền được chấp thuận. Tiền được hoàn về phương thức thanh toán ban đầu...",
      "page_number": 8
    }
  ]
}
```

Phản hồi đầy đủ trong [pdf_evaluation_result.json](pdf_evaluation_result.json) có ba nguồn; câu trả lời sử dụng `[chunk_51]` ở trang 8. Trên Streamlit, hộp nguồn này có nhãn `[chunk_51] sample.pdf | Trang: 8`.

Nguồn TXT trả `page_number: null` và không thêm số trang vào nhãn. API dùng `.get()` nên vẫn xử lý được index TXT cũ thiếu trường này. ID child có thể thay đổi nếu thay tài liệu hoặc cách chia chunk; cần đối chiếu ID và trang trong index của lần chạy tương ứng.

### Mã trạng thái của `/ask`

| HTTP status | Trường hợp |
| --- | --- |
| `200` | Xử lý thành công, bao gồm cả phản hồi không đủ thông tin hoặc không có đoạn đạt ngưỡng |
| `400` | `query` rỗng hoặc chỉ chứa khoảng trắng |
| `422` | Request không hợp lệ theo schema, ví dụ thiếu trường `query` |
| `503` | Không kết nối được Ollama khi cần gọi sinh câu trả lời |
| `504` | Quá thời gian chờ phản hồi Ollama |

Nếu chính FastAPI đã dừng, UI nhận lỗi kết nối HTTP, không nhận được response `503` từ backend. Các lỗi ngoài hai nhóm kết nối và timeout Ollama hiện chưa được ánh xạ riêng trong API.

## Kiểm tra cài đặt và chạy thử

### Cài mới bản TXT

Trước khi bổ sung PDF, bản TXT đã được thử bằng một `.venv` mới từ bản clone GitHub trên cùng máy:

- Windows, Python 3.12.10, PyTorch `2.15.0+cpu`.
- `pip check` không phát hiện dependency bị lỗi; các module chính import được.
- Tạo lại index được 37 chunk và 37 vector, mỗi vector 384 chiều.
- FastAPI `/health` trả về 37 chunk.
- Đã thử thủ công Streamlit → API → Ollama với câu có thông tin và câu thiếu thông tin.

Ollama và model đã có sẵn trên máy. Đây là kiểm tra môi trường Python mới trên cùng máy; chưa xác nhận lại toàn bộ tính năng PDF bằng một lần cài mới từ GitHub.

### Chạy thử PDF 13 trang

- Đọc được văn bản trên cả 13 trang của `sample.pdf` và giữ đúng số trang vật lý.
- Index đã lưu có 13 parent, 84 child chunk và mảng vector `float32` kích thước `84 × 384`.
- API đã được chạy với index PDF; `/health` trả về 84 chunk.
- Bốn phản hồi API thực tế được lưu thủ công trong [pdf_evaluation_result.json](pdf_evaluation_result.json), gồm câu hỏi, câu trả lời và toàn bộ nguồn đã đưa vào context.
- Cả 12 bản ghi nguồn trong file kết quả khớp với index về tên file, số trang và nội dung parent; các ID được trích dẫn đều thuộc context của câu tương ứng.
- Việc truyền metadata qua context, API và nhãn UI cũng đã được kiểm tra với dữ liệu PDF, TXT có số trang `None` và bản ghi TXT cũ thiếu trường trang.

| Câu hỏi | Nội dung trả lời đã đối chiếu thủ công | Trích dẫn | Trang |
| --- | --- | --- | --- |
| Khoản hoàn tiền được xử lý trong bao lâu và tính từ mốc nào? | 5 ngày làm việc kể từ ngày phương án hoàn tiền được chấp thuận | `[chunk_51]` | 8 |
| Đơn nội thành xác nhận đúng 15 giờ bắt đầu tính thời gian giao từ ngày nào? | Ngày làm việc kế tiếp, theo quy định từ 15 giờ trở đi | `[chunk_17]` | 3 |
| Địa chỉ email hỗ trợ của cửa hàng là gì? | `hotro@cuahangmau.example` | `[chunk_80]` | 13 |
| Số điện thoại hỗ trợ của cửa hàng là gì? | Tài liệu không công bố số điện thoại hỗ trợ | `[chunk_84]` | 13 |

Cả **4/4 ca này** có câu trả lời phù hợp với nội dung nguồn và trích đúng trang. Ở câu hỏi số điện thoại, mô hình nêu giới hạn thông tin của tài liệu, không tự tạo số điện thoại.

`pdf_evaluation_result.json` là snapshot lưu thủ công từ các lần hỏi ban đầu. Bộ đánh giá tự động dưới đây dùng cùng bốn câu hỏi và ghi báo cáo riêng. Bốn ca kiểm tra trên một tài liệu giả lập chưa đo được chất lượng tổng quát của ứng dụng hoặc khả năng xử lý mọi PDF dài.

Phiên bản PDF cũng đã được thử trên bản clone GitHub có sẵn trên cùng máy: cập nhật code và thư viện, tạo lại 84 chunk/vector, nạp API và hỏi qua Streamlit về hoàn tiền cùng số điện thoại. Lần kiểm tra này tái sử dụng `.venv` của bản clone, bổ sung phụ thuộc mới; đây chưa phải một lần tạo môi trường Python hoàn toàn mới cho bản PDF.

## Đánh giá PDF qua API

Bộ câu hỏi nằm trong [pdf_evaluation_cases.py](pdf_evaluation_cases.py). Mỗi ca khai báo `query`, `expected_fact_groups`, `expected_source` và `expected_page`.

Trước khi chạy, tạo index PDF theo hướng dẫn bên trên, chọn `INDEX_DIR = "index_pdf"` trong `config.py`, rồi khởi động API. Kiểm tra `/health` có `chunk_count: 84` với dữ liệu mẫu. Giữ Ollama và FastAPI hoạt động, mở terminal khác trong thư mục dự án và chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 pdf_evaluation.py
```

[pdf_evaluation.py](pdf_evaluation.py) lần lượt gửi từng câu hỏi tới `POST http://127.0.0.1:8000/ask` và kiểm tra phản hồi của backend đang chạy. Script dùng index đã được API nạp; việc chọn index và tạo embedding tài liệu được thực hiện ở các bước chuẩn bị trước đó.

### Nhóm cụm từ mong đợi

`expected_fact_groups` là danh sách các nhóm; mỗi nhóm chứa một hoặc nhiều cụm diễn đạt cùng một ý. Quy tắc là **OR trong nhóm, AND giữa các nhóm**: một nhóm khớp khi có ít nhất một phương án xuất hiện, và câu trả lời phải khớp tất cả các nhóm bắt buộc.

Ví dụ, câu hoàn tiền cần đủ thời hạn và mốc bắt đầu:

```python
expected_fact_groups = [
    ["5 ngày làm việc"],
    ["ngày phương án hoàn tiền được chấp thuận"]
]
```

Câu số điện thoại chấp nhận hai cách diễn đạt của cùng một ý:

```python
expected_fact_groups = [
    ["không được công bố", "không công bố"]
]
```

Cách chấm này được bổ sung sau khi câu trả lời đúng về nghĩa “Tài liệu không công bố số điện thoại...” từng bị chấm trượt vì cụm cố định là “không được công bố”. Các phương án được khai báo trong bộ câu hỏi và được chuyển thành chữ thường, gộp khoảng trắng trước khi so khớp với câu trả lời.

### Điều kiện chấm từng ca

| Trường trong báo cáo | Cách kiểm tra |
| --- | --- |
| `missing_facts` | Các nhóm trong `expected_fact_groups` chưa khớp phương án nào; mỗi phần tử là cả một nhóm, nên kết quả có dạng danh sách lồng nhau |
| `invalid_ids` | Các ID dạng `[chunk_ID]` được trích nhưng không thuộc `retrieved_sources` của câu đó |
| `source_match` | Có ít nhất một nguồn **được câu trả lời trích dẫn**, khớp đồng thời tên file và số trang mong đợi trên cùng bản ghi |
| `passed` | Với ca chấm được phản hồi: `missing_facts` rỗng, `invalid_ids` rỗng và `source_match` là `True`; ca gặp lỗi request được đặt là `False` |
| `error` | `null` khi ca đã xử lý được phản hồi; chuỗi mô tả ngoại lệ khi bắt được `requests.exceptions.RequestException` |

Trang mong đợi chỉ xuất hiện trong context chưa đủ để đạt `source_match`; câu trả lời phải trích ID của nguồn đó. Câu trả lời không có trích dẫn cũng sẽ có `source_match: false`.

`missing_facts: []` nghĩa là mọi nhóm thông tin bắt buộc đã khớp. Nếu câu hoàn tiền chỉ nêu thời hạn, kết quả thiếu là `[["ngày phương án hoàn tiền được chấp thuận"]]`.

### Xử lý lỗi từng ca

Mỗi lượt gọi và chấm được đặt trong `try/except`. Nếu bắt được `requests.exceptions.RequestException`, như lỗi kết nối, timeout, HTTP hoặc lỗi giải mã JSON của Requests, script thêm bản ghi lỗi rồi tiếp tục câu hỏi kế tiếp.

Trong bản ghi lỗi, `answer` là `null`, `retrieved_sources` là `[]`, `passed` là `false` và `error` chứa chuỗi mô tả ngoại lệ. Các trường `missing_facts`, `invalid_ids` và `source_match` là `null`, biểu thị các phép kiểm tra chưa được thực hiện. Với ca xử lý được phản hồi, `error` là `null`; ca đó vẫn có thể trượt điều kiện chấm.

### Báo cáo và kết quả đã ghi nhận

Sau khi duyệt xong các ca, script ghi một lần bằng UTF-8 vào [pdf_evaluation_report.json](pdf_evaluation_report.json), đặt cạnh script. Mỗi bản ghi lưu `query`, `answer`, toàn bộ `retrieved_sources`, `missing_facts`, `invalid_ids`, `source_match`, `passed` và `error`. Với ca chấm được phản hồi, tập ID không hợp lệ được chuyển thành danh sách trước khi ghi JSON.

| Nội dung kiểm tra | Kết quả của lần chạy đã lưu |
| --- | --- |
| Tổng số ca | 4 |
| Ca đạt đồng thời ba điều kiện | 4/4 |
| Ca có lỗi request được ghi nhận | 0/4 |
| Ca thiếu nhóm thông tin mong đợi | 0/4 |
| Ca có ID trích dẫn ngoài context | 0/4 |
| Ca trích đúng tên file và trang mong đợi | 4/4 |

Các nguồn được trích trong báo cáo là `[chunk_51]` ở trang 8 cho hoàn tiền, `[chunk_17]` ở trang 3 cho mốc xác nhận đúng 15 giờ, `[chunk_80]` ở trang 13 cho email và `[chunk_84]` ở trang 13 cho thông tin số điện thoại không được công bố.

Mỗi lần duyệt xong bộ câu hỏi sẽ **ghi đè báo cáo tự động** bằng kết quả mới, kể cả khi mọi ca đều gặp `RequestException`. Ngoại lệ khác, như `KeyError` hoặc `TypeError`, vẫn có thể dừng chương trình; nếu lỗi xảy ra trước bước ghi file, báo cáo cũ vẫn còn. Cần xác nhận lần chạy đã hoàn thành khi đọc kết quả. `evaluation_report.py` vẫn đọc kết quả TXT; báo cáo PDF được tạo trực tiếp bởi `pdf_evaluation.py`.

Đây là bốn ca phát triển trên một PDF giả lập, đã được dùng để điều chỉnh bộ chấm. `passed` phản ánh việc khớp các phương án cụm từ được khai báo và metadata nguồn. Cách diễn đạt chưa có trong nhóm vẫn có thể bị chấm trượt; việc khớp chuỗi cũng chưa xác nhận đầy đủ ý nghĩa của câu trả lời. Phép kiểm tra nguồn xác nhận ID, tên file và trang, chưa tự đánh giá từng phát biểu có được nội dung nguồn hỗ trợ hay không.

## Đánh giá trên bộ TXT

Bộ câu hỏi nằm trong [policy_evaluation_cases.py](policy_evaluation_cases.py). Bật Ollama rồi chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 policy_evaluation.py
.\.venv\Scripts\python.exe -X utf8 evaluation_report.py
```

Evaluator hiện **đọc TXT và tạo lại embedding trong bộ nhớ**, sau đó gọi trực tiếp pipeline. Nó không dùng index đã lưu, không chọn nguồn theo `INDEX_DIR` và không gửi request qua FastAPI hoặc Streamlit. Lệnh đầu ghi đè file kết quả tương ứng với cấu hình và số ca hiện tại.

Snapshot TXT hiện có: [evaluation_policy_rerank_k16_n8.json](evaluation_policy_rerank_k16_n8.json), cấu hình `k=16`, `min_score=0.3`, `top_n=3`. Các kết quả bên dưới không phải kết quả đánh giá PDF.

`evaluation_report.py` hiện đọc cố định file `evaluation_policy_rerank_k16_n8.json`. Khi đổi `k` hoặc số ca đánh giá, cần cập nhật đường dẫn đầu vào của report để đọc đúng file kết quả mới.

| Nội dung kiểm tra | Kết quả |
| --- | --- |
| Tổng số câu hỏi | 8 |
| Câu có thông tin trả lời trong tài liệu | 6 |
| Câu có chứa cụm đáp án mong đợi, không phân biệt hoa/thường | 6/6 |
| Câu thiếu thông tin | 2 |
| Câu thiếu thông tin được đọc và xác nhận từ chối phù hợp | 2/2, kiểm tra thủ công |
| Câu có ID trích dẫn ngoài context đã chọn | 0/8 |

Đây là bộ câu hỏi nhỏ dùng trong quá trình phát triển và điều chỉnh hệ thống, chưa phải bộ test độc lập. So khớp chuỗi không đánh giá đầy đủ ý nghĩa, tính đầy đủ hay độ chính xác của toàn bộ câu trả lời. ID trích dẫn nằm trong context cũng chưa chứng minh nội dung được trích dẫn hỗ trợ đúng phát biểu. Các con số trên không phải độ chính xác tổng quát của ứng dụng.

### Quan sát khi điều chỉnh retrieval

Với câu hỏi **“Điều kiện để đổi sản phẩm trong 14 ngày là gì?”**, thử nghiệm `k=12` từng bỏ sót mục 5 — phần chứa điều kiện chấp thuận đổi. Tăng `top_n` ở reranker không khôi phục được phần đã bị loại ở bước lấy ứng viên. Khi tăng `k` lên 16, một child của mục 5 được đưa vào và parent của nó xuất hiện trong context cuối.

Thử nghiệm này cho thấy cần kiểm tra cả danh sách ứng viên trước reranking. Giá trị `k=16` là lựa chọn hiện tại trên dữ liệu thực hành, chưa được chứng minh tối ưu cho tài liệu hay cách diễn đạt khác.

## Tài liệu dài

Pipeline có thể lập index nhiều chunk và chọn phần liên quan để trả lời câu hỏi. Toàn bộ tài liệu được dùng khi tạo index; mỗi lần hỏi chỉ chọn một phần nội dung làm context. Dự án đã chạy thử PDF có lớp văn bản gồm **13 trang, 84 child chunk**, với bốn câu hỏi được kiểm tra tự động qua API và đọc đối chiếu nguồn. Chưa đo giới hạn số trang, hiệu năng trên dữ liệu lớn hoặc chất lượng trên tài liệu dài và bố cục phức tạp hơn.

Các điểm cần kiểm tra khi mở rộng dữ liệu:

- Parent PDF hiện là toàn trang. Reranker đang dùng khai báo giới hạn 512 token đầu vào cho cặp câu hỏi và nội dung; chuỗi quá dài sẽ bị cắt. Vì vậy, tìm được child liên quan chưa bảo đảm reranker đọc được toàn bộ trang chứa child đó. Xem [cấu hình tokenizer của model](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/blob/main/tokenizer_config.json) và [tài liệu CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
- Context builder nối các parent được chọn, nhưng client chưa tính ngân sách token của toàn prompt. Dữ liệu đưa vào mô hình cần phù hợp với [cửa sổ ngữ cảnh được cấu hình trong Ollama](https://docs.ollama.com/context-length).
- Mỗi câu hỏi hiện chọn tối đa 3 parent. Câu hỏi cần tổng hợp nhiều trang có thể thiếu căn cứ; chưa có pipeline riêng để tóm tắt toàn bộ tài liệu dài.
- Truy xuất quét tuyến tính mọi child vector và giữ index trong RAM. Chưa đo thời gian xử lý hoặc mức sử dụng bộ nhớ khi số chunk tăng lớn.

## Các file chính

| File | Vai trò |
| --- | --- |
| [build_index.py](build_index.py) | Tạo index TXT trong `index/` |
| [pdf_index_demo.py](pdf_index_demo.py) | Tạo và lưu index PDF mẫu trong `index_pdf/` |
| [document_loader.py](document_loader.py) | Chọn loader theo đuôi file và chuẩn hóa cấu trúc đầu ra |
| [load_documents.py](load_documents.py), [pdf_loader.py](pdf_loader.py) | Đọc đoạn TXT hoặc văn bản từng trang PDF |
| [chunking.py](chunking.py) | Chia văn bản thành các chunk theo token |
| [knowledge_base.py](knowledge_base.py), [index_store.py](index_store.py) | Tạo records/vector, lưu và nạp index |
| [retrieval.py](retrieval.py), [semantic_retrieval.py](semantic_retrieval.py) | Tính cosine và chọn child chunk |
| [reranking.py](reranking.py) | Chấm lại và sắp xếp các parent ứng viên |
| [context_builder.py](context_builder.py), [prompt_builder.py](prompt_builder.py) | Ghép nguồn và xây prompt |
| [llm_client.py](llm_client.py) | Gọi Ollama |
| [rag_pipeline.py](rag_pipeline.py) | Điều phối các bước hỏi đáp |
| [config.py](config.py) | Chọn index, tham số truy xuất và số kết quả rerank |
| [api_server.py](api_server.py), [ui.py](ui.py), [app.py](app.py) | Backend, giao diện web và CLI |
| [policy_evaluation.py](policy_evaluation.py), [evaluation.py](evaluation.py), [evaluation_report.py](evaluation_report.py) | Chạy đánh giá, lấy ID trích dẫn và tổng hợp kết quả |
| [pdf_evaluation_result.json](pdf_evaluation_result.json) | Lưu thủ công bốn phản hồi API trên PDF 13 trang, kèm nội dung nguồn và số trang |
| [pdf_evaluation_cases.py](pdf_evaluation_cases.py) | Khai báo bốn câu hỏi PDF, các nhóm cách diễn đạt và tên file/trang nguồn mong đợi |
| [pdf_evaluation.py](pdf_evaluation.py) | Gọi API, chấm nhóm thông tin và trích dẫn, ghi lỗi từng ca và báo cáo JSON |
| [pdf_evaluation_report.json](pdf_evaluation_report.json) | Kết quả đánh giá PDF, gồm phản hồi, điều kiện đạt/trượt và trường lỗi request |

Các file như `embedding_demo.py`, `tokenizer_demo.py`, `rerank_demo.py` và `knowledge.txt` lưu những bài thực hành trước đó. [pdf_demo.py](pdf_demo.py) là bài thử trích xuất văn bản PDF; việc tạo index PDF được thực hiện bằng `pdf_index_demo.py`.

## Giới hạn hiện tại

- Mỗi script build đang có một đường dẫn tài liệu được khai báo trong code. Mỗi runtime đọc một index được chọn; chưa có upload, gom nhiều tài liệu hoặc tìm kiếm đồng thời nhiều index.
- PDF hiện hỗ trợ trích xuất lớp văn bản. Chưa có OCR cho trang chỉ chứa ảnh; bố cục nhiều cột và bảng biểu chưa được đánh giá. Xem [giới hạn trích xuất văn bản của pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html).
- Retrieval quét tuyến tính toàn bộ vector bằng cosine; index JSON/NumPy chưa có tìm kiếm ANN hay vector database.
- Chưa tự phát hiện index cũ; cần tự đảm bảo embedding model khi hỏi khớp với model đã dùng để build.
- Chưa đo độ trễ, tải đồng thời hoặc chất lượng trên bộ câu hỏi độc lập lớn hơn.
- Prompt hướng dẫn trích dẫn và từ chối khi thiếu dữ liệu; API chưa tự kiểm chứng ngữ nghĩa câu trả lời hay tính hợp lệ của citation trước khi trả về.
- Phiên bản hiện tại phục vụ chạy local; chưa có xác thực người dùng, đóng gói triển khai hoặc streaming.

## Hướng phát triển

- Khóa đầy đủ dependency, phiên bản PyTorch và ghi nhận phiên bản model; thử cài mới lại bản có hỗ trợ PDF.
- Mở rộng bộ đánh giá với câu diễn đạt lại, câu cần nhiều mục và câu thiếu thông tin; tách bộ phát triển và bộ test.
- Mở rộng bộ PDF sang tài liệu dài hơn, câu hỏi cần nhiều trang và các cách diễn đạt khác.
- Bổ sung thời điểm chạy, cấu hình, phiên bản model và phân loại lỗi vào báo cáo đánh giá.
- Giới hạn kích thước parent và tổng token context để xử lý tài liệu dài phù hợp với từng mô hình.
- Đo chất lượng retrieval, độ đầy đủ của câu trả lời và thời gian xử lý để so sánh cấu hình.
- Thêm thông tin phiên bản index để phát hiện thay đổi tài liệu hoặc embedding model trước khi hỏi đáp.
