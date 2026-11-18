# Danh Sách Kiểm Tra Thành Phần VLM Kiến Trúc Lai (Hybrid VLM Component Checklist)

Tài liệu này theo dõi và đối chiếu các thành phần trong dự án theo định hướng kiến trúc lai (**Hybrid VLM - LLaVA Style**):
- **Tự xây dựng từ đầu (From Scratch):** Toàn bộ nhánh Vision Tower (Vision Transformer) và Vision-Language Projector.
- **Kế thừa từ Pretrained LLM (Pretrained & Frozen):** Tokenizer, Text Token Embedding, LLM Decoder Backbone và LM Head (ví dụ từ `Qwen/Qwen2.5-0.5B` hoặc `meta-llama/Llama-3.2-1B`).
- **Mở rộng tương lai (Future Extension):** Tích hợp kiến trúc Mixture of Experts (MoE).

> **Quy ước trạng thái:**
> - `[x]` Đã xây dựng thành phần cơ bản (trong `src/`).
> - `[~]` Đã có một phần / cần tự rà soát hoặc điều chỉnh cấu trúc trước khi ghép nối.
> - `[ ]` Chưa xây dựng.

---

## 1. Tổng Quan Hiện Trạng Mã Nguồn Dự Án

- `src/common/visual_encoder.py` (`VisualEncoder`): Thực hiện phép chiếu tích chập 2D (`kernel_size=16`, `stride=16`) để cắt ảnh thành các patch và làm phẳng thành chuỗi vector `(Batch, Num_Patches, d_vision)`.
- `src/common/attention.py` (`MultiHeadAttention`): Cài đặt Scaled Dot-Product Multi-Head Attention, có hỗ trợ bật/tắt Causal Mask qua cờ `is_decoded`.
- `src/encoderblock.py` (`EncoderBlock`): Cài đặt cấu trúc Transformer Encoder (Pre-Norm, Multi-Head Attention, Residual, MLP/GELU, Residual), hiện đang chứa trực tiếp `VisualEncoder` ở đầu block.
- `test.ipynb`: Sổ tay thử nghiệm (hiện đang trống).

---

## 2. Phân Nhóm Kiểm Tra Chi Tiết Theo Kiến Trúc Mới

### Nhóm 1: Nhánh Thị Giác — Vision Tower (Tự Xây Dựng Từ Đầu)
- [x] **Patch Embedding** (`src/common/visual_encoder.py` - `VisualEncoder`): Chia ảnh thành chuỗi patch tokens kích thước $d_{\text{vision}}$.
- [x] **Bidirectional Multi-Head Self-Attention** (`src/common/attention.py` - `MultiHeadAttention` với `is_decoded=False`): Cho phép các patch ảnh tương tác ngữ cảnh hai chiều.
- [~] **Transformer Encoder Block** (`src/encoderblock.py` - `EncoderBlock`): Đã có đủ Pre-Norm, Attention, MLP (GELU) và Residual.
  - *Điểm kiến trúc cần rà soát:* Tách lớp cắt patch (`VisualEncoder`) ra khỏi thân `EncoderBlock` để Patch Embedding chỉ chạy 1 lần ở đầu vào, cho phép xếp chồng $N$ tầng Encoder Block liên tiếp.
  - *Điểm tham số cần rà soát:* Đồng bộ `block_size` với tổng số patch ảnh thực tế ($N_p = \frac{H}{P} \times \frac{W}{P}$).
- [ ] **Positional Embedding cho ảnh**: Vector mã hóa vị trí không gian (1D hoặc 2D) cộng vào chuỗi patch tokens để mô hình nhận biết tọa độ các mảnh ảnh.
- [ ] **Vision Encoder hoàn chỉnh (Backbone)**: Đóng gói toàn bộ luồng `Patch Embedding -> (+ Positional Embedding) -> N × Encoder Block -> Final LayerNorm`.

---

### Nhóm 2: Cầu Nối Đa Phương Thức — Multimodal Projector (Tự Xây Dựng Từ Đầu)
- [ ] **Multimodal Projector (Linear hoặc 2-layer MLP)**:
  - Khối ánh xạ chiều vector từ không gian thị giác sang không gian ngôn ngữ: $d_{\text{vision}} \to d_{\text{llm}}$ (với $d_{\text{llm}}$ là chiều ẩn của Pretrained LLM bạn chọn, ví dụ 896 cho Qwen2.5-0.5B).
  - Cấu trúc 2 tầng Linear xen kẽ hàm kích hoạt phi tuyến (GELU/SiLU) để tăng khả năng biểu đạt căn chỉnh ngữ nghĩa.
- [ ] **(Tùy chọn nâng cao) Token Pooling / Compression**: Cơ chế nén số lượng patch token (nếu dùng ảnh độ phân giải cao).

---

### Nhóm 3: Nhánh Văn Bản & Mô Hình Ngôn Ngữ (Kế Thừa Pretrained LLM)
*Nhờ việc sử dụng Pretrained LLM, bạn **không** cần phải tự huấn luyện BPE Tokenizer hay tự pretrain một LLM từ đầu.*

- [ ] **Pretrained Tokenizer (Kế thừa)**: Tải và sử dụng Tokenizer đi kèm với LLM mục tiêu (ví dụ từ Hugging Face: Qwen hoặc Llama) để biến câu hỏi thành danh sách `input_ids`.
- [ ] **Special Tokens**: Đăng ký hoặc sử dụng token đặc biệt đại diện cho vị trí hình ảnh (ví dụ `<image>`).
- [ ] **Pretrained Text Embedding Layer (Kế thừa)**: Tận dụng tầng nhúng từ của LLM để chuyển `input_ids` thành vector biểu diễn $(B, N_t, d_{\text{llm}})$.
- [ ] **Pretrained LLM Decoder Backbone (Kế thừa & Đóng băng)**:
  - Sử dụng toàn bộ khối Decoder có sẵn của mô hình ngôn ngữ mục tiêu.
  - Đóng băng toàn bộ trọng số (không tính gradient) trong giai đoạn huấn luyện đầu để bảo toàn ngữ pháp và tri thức ngôn ngữ tự nhiên.
- [ ] **Pretrained LM Head (Kế thừa)**: Tầng chiếu cuối cùng từ $d_{\text{llm}}$ sang kích thước từ vựng `vocab_size` để sinh logits dự đoán token tiếp theo.

---

### Nhóm 4: Hợp Nhất Đa Phương Thức & Vòng Lặp Suy Luận (Tự Lắp Ráp)
- [ ] **Multimodal Sequence Fusion**:
  - Cơ chế ghép chuỗi đặc trưng ảnh sau Projector $(B, N_p, d_{\text{llm}})$ và chuỗi nhúng văn bản $(B, N_t, d_{\text{llm}})$ thành chuỗi tensor duy nhất $(B, N_p + N_t, d_{\text{llm}})$.
  - Truyền trực tiếp tensor chuỗi hợp nhất vào Pretrained LLM qua tham số nhận vector nhúng (`inputs_embeds`).
- [ ] **Attention Mask & Causal Generation**: Đảm bảo cấu hình Attention Mask hợp lệ cho chuỗi đa phương thức.
- [ ] **Autoregressive Generation Loop**:
  - Giai đoạn nạp ngữ cảnh (Prefill) với chuỗi ảnh + câu hỏi.
  - Vòng lặp sinh từng từ tiếp theo (Decode loop) với các chiến lược: Greedy, Temperature Scaling, Top-K, Top-P Sampling cho đến khi gặp token kết thúc câu (`<eos>`).

---

### Nhóm 5: Dữ Liệu & Chiến Lược Huấn Luyện (Training Pipeline)
- [ ] **Image Preprocessing**: Resize ảnh về kích thước cố định chia hết cho `patch_size`, chuẩn hóa Mean/Std.
- [ ] **Multimodal Dataset & Collate**: Quản lý cặp dữ liệu (Ảnh, Prompt, Câu trả lời), tự động tạo batch và padding.
- [ ] **Loss Masking**: Cơ chế đánh dấu bỏ qua (ignore index) không tính Cross-Entropy loss trên phần Visual Tokens, phần câu hỏi (prompt) và các vị trí padding; chỉ tính loss trên các token của câu trả lời.
- [ ] **Chiến lược Huấn luyện 2 bước**:
  - *Bước 1 (Feature Alignment):* Đóng băng hoàn toàn Pretrained LLM và Vision Encoder, **chỉ huấn luyện duy nhất Multimodal Projector** trên tập cặp ảnh-mô tả (Captioning).
  - *Bước 2 (Visual Instruction Tuning - Tùy chọn):* Mở khóa thêm Vision Encoder (hoặc dùng LoRA cho LLM) để tinh chỉnh trên tập dữ liệu hỏi đáp đa phương thức (VQA).
- [ ] **Kiểm thử trên notebook**: Viết kịch bản kiểm tra kích thước các tensor (shape verification) tại [test.ipynb](file:///d:/Huy/AI_VIETNAM/VLM_from_scratch/test.ipynb).

---

## 3. Lộ Trình Mở Rộng Tương Lai: Tích Hợp Mixture of Experts (MoE)

Khi mô hình VLM cơ bản đã chạy thông suốt từ đầu đến cuối, việc tích hợp MoE có thể thực hiện theo 2 hướng:

### Hướng A: Tự xây dựng tầng MoE cho Vision Encoder (Vision-MoE)
- [ ] **Router / Gating Network**: Mạng tuyến tính chấm điểm phân bổ cho từng patch token ảnh tới $E$ chuyên gia.
- [ ] **Top-K Routing**: Chỉ kích hoạt $K$ chuyên gia cho mỗi patch token ($K \ll E$).
- [ ] **Expert FFNs**: Tập hợp các khối MLP độc lập thay thế cho Dense MLP trong `EncoderBlock`.
- [ ] **Weighted Sum Output**: Tổng hợp đầu ra có trọng số từ các chuyên gia được chọn.
- [ ] **Auxiliary Load Balancing Loss**: Hàm mất mát phụ giúp phân bổ patch đồng đều giữa các chuyên gia.

### Hướng B: Nâng cấp phần Pretrained LLM sang Pretrained MoE Model
- [ ] Thay thế Decoder Dense (ví dụ Qwen2.5-Dense) bằng một Pretrained MoE nhỏ gọn (như `Qwen1.5-MoE-A2.7B` hoặc các biến thể MoE mã nguồn mở tương thích), vẫn giữ nguyên quy trình kết nối với Vision Encoder và Projector tự viết.
