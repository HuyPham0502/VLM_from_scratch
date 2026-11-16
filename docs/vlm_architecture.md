# Kiến Trúc Vision-Language Model (VLM) & Lộ Trình Tích Hợp Mixture of Experts (MoE)

Tài liệu này hệ thống hóa nền tảng kiến trúc của một **Vision-Language Model (VLM)** xây dựng từ đầu, luồng biến đổi dữ liệu giữa thị giác và ngôn ngữ, cơ chế huấn luyện/suy luận, và thiết kế mở rộng sang mô hình **Sparse Mixture of Experts (MoE)** trong tương lai.

---

## 1. Sơ Đồ Kiến Trúc Tổng Quát

![Kiến trúc VLM tổng quát và phần mở rộng MoE](./vlm_moe_architecture.jpg)

Một VLM hiện đại theo hướng **Decoder-only + Vision-Language Connector** (phổ biến ở các họ mô hình như LLaVA, Qwen-VL, InternVL) gồm 4 trụ cột chính:
1. **Vision Tower (Vision Encoder):** Trích xuất đặc trưng ngữ nghĩa của ảnh dưới dạng chuỗi **Visual Tokens**.
2. **Vision-Language Projector (Connector):** Ánh xạ chuỗi Visual Tokens từ không gian thị giác sang cùng không gian biểu diễn của mô hình ngôn ngữ.
3. **Language Model Backbone (LLM Decoder):** Tiếp nhận chuỗi hợp nhất gồm Visual Tokens và Text Tokens để hiểu ngữ cảnh đa phương thức và sinh văn bản tự hồi quy (autoregressive).
4. **Cải tiến tương lai — Sparse MoE Layer:** Thay thế các khối Feed-Forward Network dày đặc (Dense FFN) bên trong Transformer bằng mạng chuyên gia thưa (Mixture of Experts) để mở rộng năng lực tri thức mà vẫn giữ chi phí tính toán thấp trên mỗi token.

> **Ghi chú về các họ kiến trúc VLM khác:** Ngoài hướng *Decoder-only + Connector* nêu trên (dễ mở rộng, dễ xây dựng từ đầu và tối ưu hiệu năng tốt nhất hiện nay), trong nghiên cứu còn có hai hướng khác là *Cross-Attention VLM* (giữ nguyên LLM và chèn thêm các tầng Cross-Attention nơi Text làm Query, Ảnh làm Key/Value như Flamingo) và *Dual-Encoder* (mã hóa ảnh và chữ song song để đối chiếu tương đồng như CLIP — chuyên cho truy xuất/phân loại hơn là sinh câu trả lời tự do).

---

## 2. Luồng Biến Đổi Dữ Liệu Tổng Thể (End-to-End Flow)

Giả sử đầu vào là một lô $B$ ảnh có kích thước $(B, C, H, W)$ và $B$ câu truy vấn văn bản:

1. **Nhánh Thị Giác (Vision Pathway):**
   - Ảnh $(B, C, H, W)$ được chia thành lưới các mảnh (patch) kích thước $P \times P$, tạo thành $N_p = \frac{H}{P} \times \frac{W}{P}$ mảnh.
   - Mỗi mảnh được chiếu thành một vector chiều $d_{\text{vision}}$, thu được chuỗi token ảnh $(B, N_p, d_{\text{vision}})$ (có thể cộng thêm 1 token `[CLS]` thành $N_p + 1$ nếu thiết kế theo chuẩn ViT cổ điển).
   - Cộng thêm **Positional Embedding** để giữ thông tin tọa độ không gian của từng mảnh ảnh.
   - Đi qua $N_{\text{vis}}$ khối **Transformer Encoder** (tương tác hai chiều giữa tất cả các patch), đầu ra vẫn có kích thước $(B, N_p, d_{\text{vision}})$.

2. **Cầu Nối Đa Phương Thức (Alignment / Projection):**
   - **Vision-Language Projector** biến đổi từng visual token từ chiều $d_{\text{vision}}$ sang chiều $d_{\text{llm}}$ của mô hình ngôn ngữ, cho ra $(B, N_p, d_{\text{llm}})$ (hoặc $(B, N'_p, d_{\text{llm}})$ với $N'_p < N_p$ nếu có bước nén token).

3. **Nhánh Ngôn Ngữ & Hợp Nhất (Text Pathway & Fusion):**
   - Văn bản đầu vào được **Tokenizer** tách thành chuỗi $N_t$ chỉ số token $(B, N_t)$.
   - Qua bảng **Token Embedding** thành $(B, N_t, d_{\text{llm}})$.
   - Chuỗi Visual Tokens và Text Tokens được ghép nối (hoặc chèn trực tiếp Visual Tokens vào vị trí token đánh dấu `<image>` trong prompt) để tạo thành chuỗi đa phương thức có tổng độ dài $T = N_p + N_t$ và kích thước $(B, T, d_{\text{llm}})$.

4. **Giải Mã Ngôn Ngữ (LLM Decoder & Output):**
   - Chuỗi đa phương thức đi qua $N_{\text{llm}}$ khối **Transformer Decoder** sử dụng **Causal Self-Attention** (mặt nạ nhân quả).
   - Đầu ra $(B, T, d_{\text{llm}})$ đi qua tầng chuẩn hóa cuối và **LM Head** để chiếu sang không gian từ vựng kích thước $V$, thu được logits $(B, T, V)$ phục vụ dự đoán token kế tiếp.

---

## 3. Phân Tích Chi Tiết Từng Khối Thành Phần

### 3.1. Vision Tower (Bộ Mã Hóa Hình Ảnh)
Dựa trên kiến trúc **Vision Transformer (ViT)**:
- **Patch Embedding:** Thay vì xử lý từng pixel rời rạc (gây bùng nổ độ dài chuỗi theo $H \times W$ với độ phức tạp $O((HW)^2)$ của Attention), ảnh được chia thành các ô vuông $P \times P$ không chồng lấp. Phép chiếu tuyến tính trên từng ô tương đương về mặt toán học với phép tích chập 2D có kích thước nhân lọc và bước nhảy cùng bằng $P$.
- **Positional Embedding:** Cơ chế Self-Attention bản chất là bất biến hoán vị (permutation-invariant) — nếu tráo vị trí các patch, tập giá trị đầu ra không đổi nếu không có thông tin vị trí. Vì vậy, mỗi patch bắt buộc được cộng thêm vector vị trí (1D learnable, 2D sinusoidal, hoặc 2D RoPE) để mô hình hiểu quan hệ "trên/dưới/trái/phải" trong ảnh.
- **Transformer Encoder Block (lặp $N_{\text{vis}}$ lần):**
  - **Pre-Norm:** Chuẩn hóa dữ liệu trước khi đi vào Attention và MLP giúp dòng gradient truyền ổn định khi xếp nhiều tầng sâu.
  - **Bidirectional Multi-Head Self-Attention:** Với $M_{\text{causal}} = 0$ (không che tương lai), mọi patch đều có thể trao đổi thông tin với toàn bộ các patch còn lại trong ảnh để xây dựng ngữ cảnh thị giác toàn cục.
  - **Feed-Forward Network (MLP):** Xử lý phi tuyến độc lập trên từng token, thường mở rộng chiều ẩn lên gấp 4 lần rồi thu về kích thước ban đầu.
  - **Residual Connection:** Cộng trực tiếp đầu vào của nhánh con vào đầu ra giúp tránh suy biến gradient.
- **Nguyên tắc phân tầng quan trọng:** Bước **Patch Embedding** chỉ thực hiện **một lần duy nhất** ở đầu vào của Vision Tower; các khối **Encoder Block** xếp chồng phía sau chỉ nhận và xuất chuỗi vector token để có thể mở rộng độ sâu $N_{\text{vis}}$ tùy ý.

### 3.2. Vision-Language Projector (Cầu Nối Đa Phương Thức)
Vision Encoder học không gian biểu diễn hình ảnh, còn LLM học không gian ngôn ngữ — hai không gian này không đồng nhất về số chiều lẫn phân phối ngữ nghĩa. Projector đóng vai trò "phiên dịch viên":
- **Linear Projector:** Một phép chiếu tuyến tính duy nhất từ $d_{\text{vision}} \to d_{\text{llm}}$. Đơn giản, ít tham số, dễ huấn luyện nhưng sức biểu đạt phi tuyến hạn chế.
- **MLP Projector:** Hai tầng tuyến tính xen kẽ hàm kích hoạt phi tuyến (như GELU/SiLU). Đây là lựa chọn cân bằng tốt nhất giữa độ đơn giản và khả năng căn chỉnh ngữ nghĩa (được sử dụng rộng rãi từ LLaVA-1.5 trở đi).
- **Token Compression Connectors (Tùy chọn nâng cao):** Khi ảnh có độ phân giải cao (tạo ra hàng nghìn patch), chi phí Attention ở LLM tăng theo bình phương độ dài chuỗi. Các kỹ thuật như *Pixel Shuffle* (gộp 4 patch liền kề thành 1 token chiều sâu gấp 4 rồi chiếu xuống), *Average/Spatial Pooling*, hoặc *Perceiver Resampler / Q-Former* (dùng tập cố định các query học được để trích xuất thông tin qua Cross-Attention) giúp nén số lượng visual tokens xuống mức gọn nhẹ hơn.

### 3.3. Nhánh Ngôn Ngữ (Tokenizer, Embedding & LLM Decoder)
- **Tokenizer & Special Tokens:** Biến đổi văn bản thô thành chuỗi số nguyên dựa trên bảng từ vựng $V$ (thường theo thuật toán BPE hoặc SentencePiece). Trong VLM, cần bổ sung các token điều khiển đặc biệt (bắt đầu câu, kết thúc câu, padding, và placeholder đánh dấu vị trí chèn ảnh).
- **Positional Encoding trong LLM:**
  - *Absolute Positional Embedding:* Cộng trực tiếp vector vị trí vào embedding đầu vào, giới hạn bởi độ dài ngữ cảnh tối đa cố định lúc huấn luyện (`block_size`).
  - *Rotary Position Embedding (RoPE):* Xoay vector Query và Key theo góc phụ thuộc vào khoảng cách vị trí tương đối giữa các token. Đây là chuẩn phổ biến trong các LLM hiện đại vì khái quát hóa tốt theo khoảng cách tương đối.
- **Transformer Decoder Block (lặp $N_{\text{llm}}$ lần):**
  - Điểm khác biệt cốt lõi so với Encoder Block là **Causal (Masked) Self-Attention**:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$
    Trong đó $M_{\text{causal}}$ mang giá trị $-\infty$ ở nửa tam giác trên (các vị trí $j > i$), ép xác suất chú ý vào các token tương lai bằng $0$. Nhờ vậy, khi huấn luyện song song toàn bộ chuỗi câu, mỗi token chỉ học cách dự đoán từ tiếp theo dựa trên quá khứ.
- **LM Head:** Chiếu vector ẩn ở bước cuối cùng sang không gian $V$ chiều. Trong nhiều kiến trúc, trọng số của LM Head được chia sẻ (weight tying) với bảng Token Embedding đầu vào để tiết kiệm tham số.

### 3.4. Hợp Nhất Đa Phương Thức & Mặt Nạ Chú Ý
Khi đưa chuỗi `[Visual Tokens | Text Tokens]` vào LLM Decoder:
- Các **Visual Tokens** đóng vai trò như "ngữ cảnh thị giác mở đầu" (visual prefix).
- Trong thiết kế thuần Causal Attention, visual token thứ $i$ chỉ thấy các visual token trước nó, còn toàn bộ **Text Tokens** phía sau đều nhìn thấy đầy đủ tất cả Visual Tokens lẫn phần văn bản đứng trước.
- Trong một số biến thể nâng cao (*Prefix-LM Attention*), vùng Visual Tokens được phép tương tác hai chiều với nhau, còn vùng Text Tokens vẫn giữ mặt nạ nhân quả.

---

## 4. Cơ Chế Huấn Luyện & Suy Luận

### 4.1. Hàm Mất Mát & Loss Masking
VLM huấn luyện theo mục tiêu **Next-Token Prediction** bằng hàm mất mát Cross-Entropy:
- Tại vị trí $t$ trong chuỗi đa phương thức, mô hình dự đoán token ở vị trí $t + 1$.
- **Loss Masking (cực kỳ quan trọng):** Không phải mọi vị trí đều được tính loss. Ta gán giá trị bỏ qua (ignore index) cho:
  1. Toàn bộ vị trí tương ứng với **Visual Tokens** (vì token ảnh là vector liên tục, không thuộc bảng từ vựng rời rạc để dự đoán).
  2. Phần **câu hỏi / chỉ dẫn (Prompt)** của người dùng (mô hình không cần học thuộc câu hỏi, chỉ cần học cách trả lời).
  3. Các vị trí **Padding** dùng để cân bằng độ dài trong batch.
- Như vậy, gradient chỉ cập nhật trọng số dựa trên chất lượng của phần **câu trả lời mục tiêu**.

### 4.2. Chiến Lược Huấn Luyện Hai Giai Đoạn
Nếu huấn luyện toàn bộ VLM từ đầu cùng lúc với trọng số ngẫu nhiên hoặc chưa đồng bộ, gradient lớn từ Projector chưa ổn định có thể phá hỏng biểu diễn của Vision Tower hoặc LLM:
- **Giai đoạn 1 — Feature Alignment (Pretraining):** Đóng băng Vision Encoder và LLM Decoder, **chỉ huấn luyện Vision-Language Projector** trên dữ liệu mô tả ảnh (Image Captioning) để căn chỉnh không gian vector ảnh khớp với không gian ngôn ngữ.
- **Giai đoạn 2 — End-to-End Instruction Tuning:** Mở khóa LLM Decoder (và tùy chọn mở khóa một phần hoặc toàn bộ Vision Encoder) cùng với Projector, huấn luyện trên dữ liệu hỏi đáp thị giác (VQA) và hội thoại đa phương thức.

### 4.3. Suy Luận Tự Hồi Quy (Autoregressive Inference)
Khi dự đoán câu trả lời mới:
1. Ảnh và câu hỏi đi qua mạng một lần để tạo biểu diễn ban đầu (**Prefill phase**).
2. Mô hình lấy phân phối xác suất tại token cuối cùng, chọn ra 1 token mới (qua Greedy, Temperature Scaling, Top-K hoặc Top-P/Nucleus Sampling).
3. Token mới được nối vào cuối chuỗi và đưa ngược lại vào mô hình để sinh token tiếp theo (**Decode phase**), lặp lại cho tới khi gặp token kết thúc hoặc chạm giới hạn độ dài.
4. **KV Cache (Tối ưu tốc độ):** Ở mỗi bước sinh mới, Key và Value của các token cũ không thay đổi, nên việc lưu lại bộ đệm Key/Value giúp tránh tính lại toàn bộ chuỗi từ đầu, giảm chi phí mỗi bước sinh từ $O(T^2)$ xuống $O(T)$.

---

## 5. Cải Tiến Trong Tương Lai: Tích Hợp Mixture of Experts (MoE)

### 5.1. Tại Sao Lại Tích Hợp MoE Vào VLM?
Trong mô hình dày đặc (**Dense Model**), mọi token — dù là token ảnh bầu trời, chữ viết OCR, công thức toán hay từ nối ngữ pháp — đều phải đi qua **toàn bộ** tham số của lớp FFN ở mỗi tầng. Điều này tạo ra xung đột đánh đổi: muốn mô hình thông minh hơn phải tăng kích thước FFN, dẫn đến chi phí tính toán (FLOPs) và độ trễ tăng tuyến tính theo số tham số.

**Sparse Mixture of Experts (MoE)** giải quyết bài toán này bằng nguyên lý **kích hoạt thưa (Sparse Conditional Computation)**:
- Tách một khối FFN lớn thành $E$ mạng FFN độc lập gọi là các **Experts** ($E_1, E_2, \dots, E_E$).
- Với mỗi token đi vào tầng MoE, chỉ có $K$ Experts phù hợp nhất ($K \ll E$, thường $K = 1$ hoặc $K = 2$) được kích hoạt tính toán.
- Nhờ đó, **tổng dung lượng tham số (Model Capacity)** tăng lên gấp nhiều lần để chứa kiến thức đa dạng (thị giác chi tiết, OCR, suy luận logic, đa ngôn ngữ...), trong khi **chi phí tính toán thực tế trên mỗi token (Active Parameters)** vẫn giữ ở mức nhỏ tương đương Dense Model.

### 5.2. Cấu Trúc Chi Tiết Của Một Tầng MoE
Trong kiến trúc Transformer của VLM, tầng **MoE Layer** sẽ **thay thế trực tiếp khối Dense FFN/MLP** bên trong các khối Transformer Decoder (hoặc cả một số khối Vision Encoder ở các kiến trúc Vision-MoE):

1. **Router / Gating Network:**
   - Nhận đầu vào là vector ẩn $x \in \mathbb{R}^{d_{\text{llm}}}$ của từng token riêng lẻ (sau bước Attention và LayerNorm).
   - Chiếu $x$ qua một phép biến đổi tuyến tính sang $E$ điểm số (logits), rồi chuẩn hóa bằng hàm Softmax để thu được phân phối xác suất $p(x) \in \mathbb{R}^E$ đánh giá mức độ phù hợp của token với từng Expert.
2. **Top-K Selection:**
   - Chọn ra $K$ chỉ số Expert có xác suất cao nhất từ $p(x)$, chuẩn hóa lại trọng số của $K$ Expert đó sao cho tổng bằng $1$ (gọi là gating weights $g_i(x)$).
3. **Expert Execution & Weighted Aggregation:**
   - Token chỉ được định tuyến đi qua $K$ Expert được chọn. Đầu ra của tầng MoE là tổng có trọng số:
     $$y = \sum_{i \in \text{Top-}K(x)} g_i(x) \cdot \text{Expert}_i(x)$$

### 5.3. Các Vấn Đề Cốt Lõi Khi Xây Dựng MoE Cho VLM
Khi tự cài đặt MoE từ đầu, cần nắm vững 3 cơ chế kỹ thuật quan trọng sau:

1. **Hiện tượng Router Collapse & Auxiliary Load Balancing Loss:**
   - *Vấn đề:* Trong quá trình huấn luyện, một vài Expert tình cờ cho loss tốt hơn ở những bước đầu sẽ được Router chọn nhiều hơn -> được cập nhật gradient nhiều hơn -> càng mạnh hơn -> cuối cùng Router dồn gần như toàn bộ token vào 1–2 Expert, khiến các Expert còn lại bị "chết" (không bao giờ được học).
   - *Giải pháp:* Bổ sung thêm **Load Balancing Loss** vào hàm mất mát tổng ($L_{\text{total}} = L_{\text{CE}} + \alpha \cdot L_{\text{balance}}$) để phạt nếu tỷ lệ token phân bổ vào mỗi Expert và tổng xác suất định tuyến bị lệch quá xa mức phân bố đều $1/E$. (Ngoài ra có thể dùng cơ chế *Auxiliary-loss-free balancing* bằng cách điều chỉnh bias động của router như các kiến trúc mới).

2. **Kiến Trúc Shared Experts (Chuyên Gia Dùng Chung):**
   - Trong bài toán VLM, có những kiến thức nền tảng cần thiết cho *mọi* token (ví dụ: quy tắc căn chỉnh ảnh-chữ cơ bản hoặc ngữ pháp cốt lõi).
   - Thay vì để các Expert lặp lại cùng một kiến thức chung đó, mô hình hiện đại (như DeepSeekMoE) tách riêng $K_s$ **Shared Experts** luôn luôn chạy cho mọi token, kết hợp với $K_r$ **Routed Experts** được chọn lọc động theo từng token.

3. **Đặc Thù Định Tuyến Đa Phương Thức (Multimodal Routing):**
   - Khi cả **Visual Tokens** và **Text Tokens** cùng đi qua một tầng MoE trong LLM Decoder:
     - Router có xu hướng tự nhiên (hoặc có thể được thiết kế ràng buộc) phân nhóm một số Expert chuyên xử lý đặc trưng không gian/kết cấu thị giác, và một số Expert khác chuyên xử lý suy luận ngôn ngữ.
     - Cần lưu ý cân bằng tải riêng hoặc chung giữa hai loại token để tránh tình trạng số lượng lớn Visual Tokens (thường chiếm đa số độ dài chuỗi đầu vào) làm quá tải (capacity overflow) các Expert chuyên dụng.

---

## 6. Đối Chiếu Với Phần Đã Xây Dựng Trong Dự Án

Hiện tại, mã nguồn trong `src/` đang đặt những viên gạch đầu tiên cho **Nhánh Thị Giác** và **Lõi Attention**:
- `src/common/visual_encoder.py` (`VisualEncoder`): Đảm nhiệm bước **Patch Embedding** đầu vào của Vision Tower.
- `src/common/attention.py` (`MultiHeadAttention`): Đảm nhiệm cơ chế **Scaled Dot-Product Multi-Head Attention**, hỗ trợ cả hai chế độ không che mặt nạ (cho Vision Encoder) và có Causal Mask (sẵn sàng cho LLM Decoder).
- `src/encoderblock.py` (`EncoderBlock`): Kết hợp một tầng **Transformer Encoder Block** (Pre-Norm, Attention, FFN/MLP, Residual). Khi mở rộng thành Vision Encoder nhiều tầng, bước Patch Embedding cần được tách ra chạy độc lập trước chuỗi các Encoder Block.

Để theo dõi cụ thể tiến độ từng module đã có và chưa có, xem chi tiết tại [component_checklist.md](./component_checklist.md).

