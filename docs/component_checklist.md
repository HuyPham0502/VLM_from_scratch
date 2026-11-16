# Danh Sách Kiểm Tra Thành Phần Xây Dựng VLM Từ Đầu (Component Checklist)

Tài liệu này tổng hợp và đối chiếu các thành phần đã có trong dự án hiện tại với toàn bộ các khối cần thiết của một **Vision-Language Model (VLM)** hoàn chỉnh xây dựng từ đầu, kèm theo lộ trình mở rộng **Mixture of Experts (MoE)** trong tương lai.

> **Quy ước trạng thái:**
> - `[x]` Đã xây dựng thành phần cơ bản (có thể cần tinh chỉnh khi ghép nối).
> - `[~]` Đã có một phần / cần tách hoặc điều chỉnh cấu trúc kiến trúc trước khi mở rộng.
> - `[ ]` Chưa xây dựng.

---

## 1. Tổng Quan Hiện Trạng Dự Án

Hiện tại dự án đã có 3 module nền tảng trong `src/` và 1 notebook thử nghiệm còn trống (`test.ipynb`):
- `src/common/visual_encoder.py` (`VisualEncoder`): Thực hiện bước cắt ảnh thành các patch kích thước cố định (`patch_size=16`) qua tích chập 2D và chuyển đổi thành chuỗi patch token `(Batch, Num_Patches, Embed_Dim)`.
- `src/common/attention.py` (`MultiHeadAttention`): Thực hiện Scaled Dot-Product Multi-Head Attention với cơ chế tách nhiều head, dropout, chiếu đầu ra và tùy chọn Causal Mask dạng tam giác dưới thông qua cờ `is_decoded`.
- `src/encoderblock.py` (`EncoderBlock`): Kết hợp `VisualEncoder` ở đầu vào với một khối Transformer Encoder theo cấu trúc Pre-Norm (LayerNorm -> Multi-Head Attention -> Residual -> LayerNorm -> MLP/GELU -> Residual).
- `test.ipynb`: Chưa có nội dung kiểm thử (0 bytes).

---

## 2. Danh Sách Kiểm Tra Thành Phần Chi Tiết

### A. Vision Tower (Bộ Mã Hóa Hình Ảnh)
- [x] **Patch Embedding** (`src/common/visual_encoder.py` - `VisualEncoder`): Cắt ảnh 2D thành chuỗi các vector đặc trưng của từng patch.
- [x] **Bidirectional Multi-Head Self-Attention** (`src/common/attention.py` - `MultiHeadAttention` với `is_decoded=False`): Cho phép các patch ảnh tương tác thông tin hai chiều với nhau.
- [~] **Transformer Encoder Block** (`src/encoderblock.py` - `EncoderBlock`): Đã có đầy đủ cấu trúc Pre-Norm, Attention, FFN (GELU) và Residual Connection.
  - *Điểm cần tự rà soát về kiến trúc:* Hiện tại lớp cắt patch (`VisualEncoder`) đang nằm bên trong `EncoderBlock`. Theo thiết kế chuẩn, cắt patch chỉ thực hiện **1 lần** ở đầu vào, còn khối Encoder Block chỉ nhận chuỗi token để có thể xếp chồng nhiều tầng (`N` layers) liên tiếp.
  - *Điểm cần tự rà soát về tham số & shape:* Tham số `block_size` mặc định ở `EncoderBlock` hiện là `16`, cần đảm bảo khớp với tổng số patch thực tế của ảnh đầu vào (ví dụ ảnh 224×224 với patch 16×16 sinh ra 196 patch). Ngoài ra, nên kiểm tra lại nhánh xử lý trường hợp ảnh đầu vào thiếu chiều Batch (tensor 3D).
- [ ] **Positional Embedding cho ảnh**: Vector mã hóa vị trí không gian (1D hoặc 2D) cộng vào chuỗi patch tokens để mô hình phân biệt vị trí các mảnh ảnh.
- [ ] **`[CLS]` Token (Tùy chọn theo chuẩn ViT)**: Token đặc biệt đại diện toàn bộ bức ảnh (nếu kiến trúc cần gom thông tin toàn cục hoặc dùng trong giai đoạn pretrain tương phản).
- [ ] **Vision Encoder hoàn chỉnh (Backbone)**: Module tổng hợp theo luồng `Patch Embedding -> (+ CLS) -> (+ Positional Embedding) -> N × Encoder Block -> Final LayerNorm`.

---

### B. Vision-Language Connector (Cầu Nối Đa Phương Thức)
- [ ] **Multimodal Projector**: Khối ánh xạ chiều đặc trưng từ không gian Vision (`d_vision`) sang không gian ngôn ngữ (`d_llm`), thường là Linear đơn hoặc MLP nhiều lớp kèm hàm kích hoạt phi tuyến.
- [ ] **Cơ chế nén Visual Tokens (Tùy chọn nâng cao)**: Spatial Pooling, Pixel Shuffle hoặc Perceiver Resampler / Q-Former để giảm số lượng token ảnh khi làm việc với ảnh độ phân giải cao.

---

### C. Text Processing & Embedding (Xử Lý Văn Bản)
- [ ] **Tokenizer & Vocabulary**: Bộ tách từ (BPE / WordPiece / SentencePiece hoặc tự định nghĩa bảng từ vựng cho bài toán thử nghiệm) ánh xạ văn bản thành chuỗi chỉ số nguyên (`token_ids`).
- [ ] **Special Multimodal Tokens**: Các token đặc biệt như token bắt đầu câu, kết thúc câu, đệm (padding), và token đánh dấu vị trí ảnh (ví dụ `<image>`).
- [ ] **Text Token Embedding**: Bảng tra cứu chuyển `token_ids` thành các vector trong không gian `d_llm`.
- [ ] **Positional Encoding cho chuỗi văn bản**: Mã hóa thứ tự từ trong câu (Absolute Positional Embedding hoặc Rotary Position Embedding - RoPE).

---

### D. Language Model / Multimodal Decoder (Bộ Giải Mã Ngôn Ngữ)
- [x] **Causal Masked Self-Attention** (`src/common/attention.py` - `MultiHeadAttention` với `is_decoded=True`): Đã có cơ chế mặt nạ tam giác dưới ngăn token phía trước nhìn thấy token tương lai.
- [ ] **Transformer Decoder Block**: Khối giải mã gồm `Pre-Norm -> Causal Self-Attention -> Residual -> Pre-Norm -> FFN/MLP -> Residual` (có thể bổ sung Cross-Attention nếu đi theo hướng Encoder-Decoder thay vì Decoder-only).
- [ ] **LLM Backbone**: Chuỗi xếp chồng `M` khối Decoder Block và lớp chuẩn hóa cuối (Final Normalization).
- [ ] **LM Head**: Tầng chiếu tuyến tính từ chiều ẩn `d_llm` về kích thước bảng từ vựng `vocab_size` để dự đoán xác suất token tiếp theo.

---

### E. Kiến Trúc VLM Tổng Thể & Suy Luận (Full VLM & Inference)
- [ ] **Multimodal Sequence Fusion**: Cơ chế hợp nhất chuỗi Visual Tokens (sau Projector) và Text Tokens (sau Embedding) thành một chuỗi thống nhất đưa vào Decoder.
- [ ] **Combined Attention Mask**: Quản lý mặt nạ kết hợp giữa padding mask và causal mask (hoặc cho phép các visual token nhìn thấy nhau hai chiều trong chuỗi tiền tố - Prefix LM, tùy lựa chọn thiết kế).
- [ ] **Autoregressive Generation Loop**: Quy trình sinh từng token nối tiếp nhau tại thời điểm suy luận (Greedy Decoding, Temperature, Top-K, Top-P Sampling) và điều kiện dừng khi gặp token kết thúc.
- [ ] **KV Cache (Tùy chọn tối ưu)**: Bộ nhớ đệm Key/Value của các token trước đó nhằm tăng tốc độ sinh văn bản.

---

### F. Dữ Liệu, Huấn Luyện & Kiểm Thử (Data, Training & Evaluation)
- [ ] **Image Preprocessing Pipeline**: Chuẩn hóa kích thước ảnh mục tiêu, đưa giá trị pixel về miền chuẩn (Mean/Std Normalization) phù hợp với `patch_size`.
- [ ] **Multimodal Dataset & Collate Function**: Quản lý cặp dữ liệu (Ảnh, Câu hỏi/Mô tả, Câu trả lời), tự động padding và tạo batch.
- [ ] **Target Labels & Loss Masking**: Cơ chế đánh dấu bỏ qua (ignore index) đối với các vị trí visual tokens, câu hỏi đầu vào (prompt) và token padding để chỉ tính hàm mất mát trên phần văn bản mô hình cần sinh.
- [ ] **Loss Function**: Hàm mất mát dự đoán token tiếp theo (Cross-Entropy Loss theo cơ chế dịch trái 1 bước giữa dự đoán và nhãn).
- [ ] **Training Pipeline**: Vòng lặp huấn luyện, Optimizer, Learning Rate Scheduler (Warmup + Decay), Gradient Clipping, và quản lý thiết bị (CPU/GPU).
- [ ] **Chiến lược huấn luyện nhiều giai đoạn**:
  - Giai đoạn 1 (Feature Alignment): Đóng băng Vision Encoder và LLM, chỉ huấn luyện Projector.
  - Giai đoạn 2 (Instruction Tuning): Mở khóa toàn bộ hoặc một phần LLM cùng Projector để huấn luyện hội thoại / trả lời câu hỏi.
- [ ] **Checkpointing & Validation**: Lưu/tải trọng số mô hình, theo dõi loss/metric trên tập kiểm định và chạy thử nghiệm kiểm tra kích thước tensor trên `test.ipynb`.

---

## 3. Lộ Trình Mở Rộng Tương Lai: Tích Hợp Mixture of Experts (MoE)

Khi mô hình VLM cơ bản (Dense Model) đã chạy thông suốt từ đầu đến cuối, các thành phần sau sẽ được thêm để chuyển đổi sang kiến trúc **Sparse MoE VLM**:

- [ ] **Router / Gating Network**: Mạng tuyến tính đánh giá từng token (ảnh hoặc chữ) và gán trọng số xác suất tới từng Expert trong tổng số `N` Experts.
- [ ] **Top-K Selection Mechanism**: Cơ chế chỉ kích hoạt `K` Experts có điểm số cao nhất trên mỗi token (ví dụ Top-1 hoặc Top-2) nhằm giữ chi phí tính toán thấp khi tăng tổng tham số.
- [ ] **Expert Pool (Tập hợp Experts)**: Nhiều khối FFN/MLP độc lập hoạt động song song bên trong tầng MoE thay cho một khối FFN duy nhất của Transformer Block.
- [ ] **Weighted Output Aggregation**: Tổng hợp có trọng số kết quả từ `K` Experts được chọn để tạo đầu ra của tầng MoE.
- [ ] **Auxiliary Load Balancing Loss**: Hàm mất mát phụ trợ giúp phân bổ token đồng đều giữa các Experts, tránh hiện tượng sụp đổ định tuyến (Router Collapse).
- [ ] **Tùy chọn nâng cao cho Multimodal MoE**:
  - *Shared Experts*: Một hoặc vài Expert luôn được kích hoạt cho mọi token nhằm học kiến thức nền chung.
  - *Modality-Aware Routing*: Cơ chế giúp Router chuyên biệt hóa một số Expert cho token thị giác và một số Expert cho token ngôn ngữ (hoặc tự nhiên để mô hình tự học phân tách).

