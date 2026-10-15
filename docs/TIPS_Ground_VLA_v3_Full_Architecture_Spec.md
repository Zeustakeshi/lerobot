# TIPS-Ground-VLA v3 — Full Architecture Specification

> **Version:** 3.0 · **Status:** Proposed architecture, implementation-ready specification · **Language:** Tiếng Việt  
> **Purpose:** Đặc tả **toàn bộ kiến trúc policy mới**, đủ chính xác để một agent triển khai mà không cần suy đoán về cấu trúc mạng, tensor, luồng thông tin, loss và hyperparameters.  
> **Scope:** Kiến trúc, objective, dữ liệu cần/không cần, huấn luyện, suy luận, các invariant kiểm chứng. **Không** mô tả code, framework, tập lệnh, đường dẫn repository, hay thủ thuật sửa một implementation hiện tại.  
> **Baseline independence:** Không kế thừa checkpoint TIPS-Ground-VLA v1/v2, TurboVLA, attention query weights hay action decoder từ các phiên bản trước. Có thể **khởi tạo encoder student từ pretrained ViT-S/16 và MiniLM**.

---

## 0. Tuyên bố thiết kế bắt buộc

### 0.1. Triết lý

**TIPS-Ground-VLA v3 là một VLA policy độc lập cho nhiều dạng thao tác robot**, không phải policy chỉ dành cho pick-and-place. Model học ba lớp thông tin khác nhau rồi đưa chúng vào action chunk decoder:

1. **WHAT:** liên hệ instruction với visual evidence bằng Alignment Head.
2. **WHERE / WITH WHAT / HOW RELEVANT:** tạo các **task-adaptive interaction regions** và grounded latent features bằng Grounding Head. Một region có thể là object, bộ phận vật thể, bề mặt, tay nắm, công cụ, vùng tiếp xúc hoặc phần cảnh hỗ trợ thao tác. **Không** bắt buộc mỗi instruction có `source` và `destination`.
3. **ROBOT CONDITION:** biểu diễn proprioception và trạng thái đang vận hành bằng State Head.

Cả ba đi vào **temporal latent fusion**, rồi tới **direct deterministic Transformer action-chunk decoder**. Không sử dụng diffusion, latent action VAE, autoregressive token sampling hoặc thêm controller học được thứ tư.

### 0.2. Các ràng buộc không được thay đổi

| Thuộc tính | Quyết định cố định cho v3 |
|---|---|
| Số specialized heads | **Chính xác 3**: Alignment, Grounding, State |
| Student vision | **Một ViT-S/16** pretrained, shared weights giữa các camera; **không** dùng DINO làm inference backbone |
| Student language | **MiniLM-L12-H384-uncased** pretrained, fine-tunable có kiểm soát |
| Các teacher training-only | TIPSv2-B/14 **vision + text**; DINOv3 ViT-L/16 **vision**; Grounding DINO **region proposal** |
| Segmentation offline | SAM 2 tùy chọn chỉ để nâng box thành mask / track video, không nằm trong student |
| Teacher fusion | Proposal/structure trước, semantic validation sau, **không hard-crop hoặc hard-mask theo TIPS** |
| Attention Alignment | Text-conditioned cross-attention, **không shared visual prior bias** |
| Attention Grounding | 8 cạnh tranh **interaction slots**, có iterative slot competition |
| Vai trò của slots | **Task-adaptive**, không gán cứng grasp/place/source/destination |
| Student latent width | **384** xuyên suốt ba heads, fusion và action decoder |
| Alignment latents | **8 × 384** |
| Grounding latents | **8 × 384** |
| State latents | **4 × 384** |
| Temporal memory | **4 × 384**, recurrent, chỉ chứa quá khứ |
| Unified latent Z | **20 × 384** |
| Action prediction horizon | **32** actions mỗi forward |
| Action execution horizon | **4** actions, cố định trong bản kiểm chứng này |
| Action decoder | **6-layer direct Transformer decoder**, một forward sinh cả chunk |
| Diffusion | **Không có, kể cả training** |
| Adaptive horizon / async execution | **Không có trong v3**; giữ hệ thống đơn giản để kiểm chứng giả thuyết |
| Camera calibration/depth | Không bắt buộc, không giả định tồn tại |
| Detection boxes/masks trong LeRobot | Không bắt buộc có; pseudo-labels tạo **offline** và được kiểm soát độ tin cậy |
| Student parameters | **Mục tiêu khoảng 80–95M, trần mềm 100M**; bắt buộc đếm thực tế sau khi thiết lập |

### 0.3. Những giả thuyết chính cần kiểm chứng

- **H1 — Complementary teachers:** TIPSv2 bổ sung semantic patch–text correspondence; DINOv3 bổ sung dense spatial consistency; Grounding DINO thêm localization candidates. Các tín hiệu này **không đồng nghĩa** và có thể hỗ trợ nhau nếu uncertainty được xử lý đúng.
- **H2 — Competitive task-adaptive slots:** slot competition và grounding supervision trực tiếp hữu ích hơn tám queries không phân vai cùng nhận một spatial prior.
- **H3 — Explicit grounded bottleneck:** chuyển regions thành pooled features **kèm tọa độ không gian** cho action decoder sử dụng tốt hơn latent attention không định vị tường minh.
- **H4 — State and temporal conditioning:** grounding phụ thuộc instruction và robot state, còn temporal memory giúp phân biệt tiến trình thao tác.
- **H5 — Simplicity-first action:** direct action-chunk prediction là baseline hợp lý để cô lập tác động của perception/attention, trước khi nghĩ tới mô hình hóa hành động phức tạp hơn.

**Chúng là research hypotheses, không phải kết luận đã chứng minh bằng benchmark.**

### 0.4. Không thuộc v3

- Không load checkpoint `20k` của model cũ.
- Không dùng shared attention-logit bias sinh từ `mean(all_phrase_maps)`.
- Không xem TIPS patch similarities là true object masks.
- Không xem DINOv3 feature affinity là grasp-point / destination ground truth.
- Không buộc 8 slots gắn cố định với source/destination.
- Không dùng Grounding DINO, DINOv3, TIPSv2 hoặc SAM 2 khi robot đang chạy.
- Không tạo một nhánh spatial action decoder thứ tư.
- Không có diffusion, DDIM, KV caching qua nhiều denoising steps, adaptive action execution, asynchronous chunk stitching.
- Không giả định camera coordinates = world coordinates khi thiếu calibration/depth.
- Không tuyên bố general manipulation nếu chỉ train trên vài thao tác tương tự nhau.

---

## 1. Toàn bộ đồ thị tính toán

### 1.1. Student policy (inference graph)

```text
      I_top,t           I_wrist,t            instruction q        state s_t
         |                   |                      |                  |
         +--------+----------+                      |                  |
                  v                                 v                  v
       PREPROCESS / MASK                    MiniLM-L12            STATE HEAD
                  |                           |                       |
                  v                           v                       v
       shared pretrained ViT-S/16       contextual text         Z_s [4,384]
                  |                       T [L,384]                |
                  v                           |                     |
        visual tokens F [588,384]            |                     |
                  |                           |                     |
                  +-------------+-------------+                     |
                                v                                   |
                      ALIGNMENT HEAD                               |
                dense phrase maps + Z_a [8,384]                    |
                                |                                   |
                                +-------------------+---------------+
                                                    v
                                        GROUNDING HEAD
                                        8 competitive slots
                                  region maps + spatial descriptors
                                        Z_g [8,384]
                                                    |
                          +-------------------------+-----------------+
                          |                         |                 |
                          v                         v                 v
                     Z_a [8,384]               Z_g [8,384]      Z_s [4,384]
                          |                         |                 |
                          +-------------------------+-----------------+
                                                    v
                                       current tokens [20,384]
                                                    + <--- memory M_t [4,384]
                                                    v
                                     2-LAYER TEMPORAL FUSION
                                                    |
                                       unified Z_t [20,384]
                                                    |
                                                    + ---> memory update M_(t+1)
                                                    |
                                                    v
                                  DIRECT ACTION TRANSFORMER DECODER
                                    32 learned action-position queries
                                                    |
                                                    v
                                        A_hat [32,D_action]
                                                    |
                                                    v
                                         EXECUTE FIRST 4 ACTIONS
                                                    |
                                       receive next observation
```

**Dependency order bắt buộc:** `State Head` và `Alignment Head` có thể chạy độc lập sau encoder; `Grounding Head` nhận output của cả hai; Fusion nhận cả ba. Tránh vòng lặp dependency vô ý.

### 1.2. Offline training-only teacher graph

```text
RGB frame + referring phrases                         LeRobot episode/task
              |                                             |
              +--------------+----------------+-------------+
              |              |                |
              v              v                v
     Frozen TIPSv2    Frozen DINOv3     Grounding DINO
     image + text     visual features    region proposals
              |              |                |
              v              v                v
       phrase heatmaps   patch affinities    boxes/scores
              |              |                |
              |              |          optional SAM 2 masks
              |              |                |
              +--------------+----------------+
                             |
                  confidence-aware label synthesis
                             |
                targets with validity/uncertainty
                             |
                             v
                  PERCEPTION AUXILIARY LOSSES
                             |
                   update STUDENT parameters
```

Teacher inference và candidate evaluation được xử lý offline khi có thể, để quá trình policy training không phải chạy nhiều foundation models cùng lúc. Không có teacher trong forward graph deploy.

### 1.3. Ba nguồn gradient không được nhập nhằng

- `L_semantic` tác động Alignment dense semantic branch và visual/text projection liên quan.
- `L_dense_spatial` tác động Grounding **pre-slot spatial adapter** và các phần student vision được phép fine-tune; không trực tiếp dạy object identity.
- `L_region` tác động **Grounding region predictor / slots / action-relevant spatial readouts** bằng pseudo-labels đáng tin cậy.
- `L_action` lan qua decoder → fusion → **cả ba heads** → encoder modules được unfreeze. Đường từ region probabilities/soft pooling tới action loss **không bị detach**.
- Mọi teacher luôn frozen. Teacher target không nhận gradient.

---

## 2. Ký hiệu và tensor contract toàn cục

| Ký hiệu | Định nghĩa | Mặc định |
|---|---|---|
| `B` | Batch size | 8 cho sequence action training |
| `C` | Số camera tối đa | 2 (`top`, `wrist`) |
| `H,W` | Kích thước ảnh student | **224 × 336** |
| `P_v` | Student ViT patch size | 16 |
| `h_v,w_v` | Student patch grid | **14 × 21** |
| `N_v` | Visual patches/camera | **294** |
| `N` | Visual patches của 2 camera | **588** |
| `d` | Student hidden dimension | **384** |
| `L` | Text token sequence length | **64** (pad/truncate) |
| `P` | Số referring phrases khả dụng | `0..6`/instruction |
| `Q_a` | Alignment latent queries | **8** |
| `Q_g` | Grounding interaction slots | **8** |
| `Q_s` | State latent queries | **4** |
| `Q` | Current latent tokens | **20** |
| `Q_m` | Historical memory tokens | **4** |
| `H_p` | Prediction/action chunk horizon | **32** |
| `H_e` | Execution steps giữa 2 lần observe | **4** |
| `D_s` | Robot-state dimension | Đọc từ schema, không hard-code |
| `D_a` | Robot action dimension | Đọc từ schema, không hard-code |
| `R_max` | Max teacher candidates / phrase / camera | **5** |
| `τ_sem` | Student dense semantic temperature | **0.07** ban đầu |
| `ε` | Numerical stabilizer | **1e-6** |

### 2.1. Interface của một observation

```text
I_t          [B, 2, 3, 224, 336]   — RGB views theo thứ tự đã xác định
camera_valid [B, 2]                — 1=present, 0=absent
patch_valid  [B, 2, 14, 21]        — valid patch / image-content fraction
text_ids     [B, 64]
text_valid   [B, 64]
phrase_spans [B, 6, 64]            — membership/validity từng phrase, khi có
phrase_valid [B, 6]
state_t      [B, D_s]              — robot-specific normalized channels
state_valid  [B, D_s]              — measurement validity
memory_t     [B, 4, 384]          — zero/reset lúc đầu episode
```

Không có bbox, mask, depth hay camera extrinsics là yêu cầu bắt buộc trong **input inference**.

### 2.2. Tensor contract đầu ra

```text
Visual features F             [B, 588, 384]
Contextual text T             [B, 64, 384]
Dense semantic map S          [B, 6, 588]  + phrase_valid
Alignment latents Z_a         [B, 8, 384]
State latents Z_s             [B, 4, 384]
Grounding slots Z_g           [B, 8, 384]
Slot region maps              [B, 8, 2, 14, 21]
Slot geometric descriptors    [B, 8, 2, D_geo]
Current latents               [B, 20, 384]
Fusion context                [B, 24, 384]
Unified latent Z_t            [B, 20, 384]
Action chunk A_hat            [B, 32, D_a]
Updated memory M_(t+1)        [B, 4, 384]
```

Trong đó `D_geo=6` trên mỗi camera theo mục 8; `phrase_valid` bảo đảm không lấy padded phrase làm nhãn. Nếu dataset có ít hơn 2 camera, camera vắng mặt phải được mask hoàn toàn, không nhân đôi ảnh còn lại để giả làm view khác.

---

## 3. Input hình ảnh: preprocessing và spatial correspondence

### 3.1. Quyết định độ phân giải

Student nhận **224×336** cho từng camera. Kích thước này chia hết cho patch 16, đồng thời chia hết cho patch 14 của TIPSv2. Vẫn nhẹ hơn độ phân giải cao; không biến ảnh 16:9 thành hình vuông 224×224 như v1.

Đối với ảnh gốc 720×1280, sử dụng **resize giữ tỷ lệ + symmetric letterbox padding** đến 224×336. Kích thước nội dung ảnh sau resize khoảng **189×336**, phần còn lại là padding dọc. Tọa độ phải đi theo biến đổi geometric này một cách nhất quán.

### 3.2. Mask nội dung ảnh

- Theo dõi phép biến đổi từ raw image coordinates sang student image coordinates, và phép biến đổi nghịch đảo để báo cáo visualization.
- Tạo validity map theo pixel; quy đổi thành `patch_valid` bằng phần diện tích thật của patch.
- Patch 100% padding không được là key/value hợp lệ cho Alignment hoặc Grounding.
- Patch chứa một phần padding có thể dùng, nhưng localization/distillation targets được trọng số theo valid fraction.
- ViT attention lên vùng padding phải được xử lý bằng valid-key masking hoặc một phương pháp tương đương để tránh padding trở thành shortcut; phần logic downstream luôn phải có patch validity mask.
- Mỗi camera có `camera_id_embedding` riêng; `top` và `wrist` không dùng chung identity embedding, nhưng **dùng chung ViT weights**.
- Không giả định các pixel cùng tọa độ từ `top` và `wrist` tương ứng cùng điểm 3D.

### 3.3. Coordinate system chuẩn

Mọi 2D spatial descriptor được biểu diễn trong **khung ảnh đã letterbox**: tọa độ patch center chuẩn hóa vào `[-1,1]`.

```
u_j = 2*(j+0.5)/21 - 1,
 v_i = 2*(i+0.5)/14 - 1.
```

Trước khi đánh giá so với object annotation trên ảnh gốc, phải dùng biến đổi coordinate tương ứng. Không thể lấy trực tiếp một (mu_x, mu_y) trong letterbox làm tọa độ pixel raw frame.

### 3.4. Augmentation được định nghĩa ở mức dữ liệu

- Photometric jitter nhẹ được phép; semantic targets/coordinates giữ nguyên.
- Spatial transforms chỉ được dùng nếu camera image, pseudo-labels và action labels vẫn **cùng tương ứng vật lý**.
- Không dùng random horizontal flip làm mặc định vì nó đổi handedness, hướng di chuyển và có thể đảo relation “left/right”.
- Không tạo dữ liệu mới bằng cách dịch chuyển riêng object trong ảnh nhưng giữ nguyên recorded action.
- Teacher và student phải nhìn cùng nội dung vật lý sau các phép đổi tọa độ hợp lệ.

---

## 4. Shared student ViT-S/16

### 4.1. Backbone

| Thuộc tính | Giá trị |
|---|---|
| Architecture | Pretrained Vision Transformer Small, patch16 |
| Transformer depth | **12** |
| Backbone hidden width | **384** |
| Self-attention heads | **6** |
| Head width | **64** |
| MLP expansion | **4×** (=1536) |
| Patch embedding | Conv/projection từ non-overlapping 16×16 RGB patches |
| Positional encoding | Learned pretrained embeddings, **2D interpolation** sang 14×21 grid |
| CLS | Có thể tồn tại trong backbone, nhưng **không đưa CLS vào dense patch tensor** |
| Output dense | **[B,2,294,384]** |
| Weight sharing | Một backbone duy nhất cho mọi camera |
| Initialization | Pretrained ViT-S/16; **không** lấy từ checkpoint v1 |

### 4.2. Công thức

Cho camera `c`:

\[
F^c = \operatorname{ViT}_{S/16}(I^c)_{\text{patch}}
\in \mathbb{R}^{294\times384}.
\]

Sau layer normalization/adapter nhẹ và camera embedding:

\[
\widetilde F^c = \mathrm{LN}(F^c + e_c).
\]

Concat theo camera token axis:

\[
F=[\widetilde F^{top};\widetilde F^{wrist}]
\in\mathbb R^{588\times384}.
\]

`F` không được flatten lại thành 14×42 rồi dùng 2D convolution chạy xuyên biên giới giữa các camera; bất kỳ spatial convolution nào đều được chạy **riêng từng camera**.

### 4.3. Nguyên tắc fine-tuning

- Không train từ random ViT initialization khi robot demonstrations ít.
- Stage 1: giữ **8 ViT blocks đầu frozen**, cho 4 blocks cuối LR rất thấp.
- Stage 2: tiếp tục freeze 8 blocks đầu; 4 blocks cuối LR nhỏ; train mới projection/heads/decoder với LR lớn hơn.
- Pretrained classifier head bị loại bỏ khỏi policy.

---

## 5. Student language encoder MiniLM

### 5.1. Backbone

| Thuộc tính | Giá trị |
|---|---|
| Checkpoint class | MiniLM-L12-H384-uncased |
| Depth | **12 transformer layers** |
| Hidden size | **384** |
| Internal self-attention heads | **12** (width 32/head) |
| Token maximum | **64** |
| Parameter size | Khoảng **33M** (theo model card) |
| Initialization | Pretrained |
| Inference cache | Tokenization + embeddings cache khi instruction không đổi |

Model card chính thức lưu ý checkpoint MiniLM này cần fine-tuning cho downstream tasks. Bản v3 không giả định pretrained MiniLM đã hiểu sẵn thao tác robot đặc thù.

### 5.2. Hai loại language context

1. `T_full`: contextual token features `[B,64,384]` từ toàn bộ instruction, bao gồm verbs, relations, direction và modifiers.
2. `T_phrase`: tối đa 6 referring/interaction phrases `[B,6,384]`, được tạo bằng masked pooling các contextual token theo spans. Một phrase có thể là object, part, tool, surface hoặc vùng đề cập, **không bắt buộc là source/destination**.

Global task vector:

\[
t_{global}=\mathrm{MaskedMean}(T_{full})\in\mathbb R^{384}.
\]

Phrase vector:

\[
t_p=\mathrm{MaskedMean}(T_{full},\mathrm{span}_p).
\]

### 5.3. Contract cho referring phrases

- Tối đa `P=6` phrases có thể định vị từ text; phrase nào không mang meaning về region có thể bị đánh dấu `phrase_valid=0`.
- Verbs/relations/directions vẫn nằm trong `T_full` và `t_global` để condition action/grounding.
- Nếu task không có phrase định vị rõ ràng (`go home`, `open gripper`), `P=0` là hợp lệ; Alignment/Grounding tiếp tục hoạt động dựa vào `t_global` và state.
- Một instruction có thể có nhiều objects; không gộp chúng thành một attention map trung bình.
- Phrase extraction không phải một learned fourth head của student; là một cách chuẩn bị/định dạng language supervision, fallback bằng full instruction khi không có spans.

---

---

## 6. Multi-teacher supervision — phân công nhiệm vụ tường minh

**Ba teacher giải quyết ba bài toán khác nhau.** Không dùng DINOv3 làm semantic detector, không dùng TIPSv2 để suy ra grasp point, không cho Grounding DINO chịu trách nhiệm dynamics hoặc state.

### 6.1. Teacher A — TIPSv2-B/14 (semantic correspondence)

| Thuộc tính | Quyết định |
|---|---|
| Teacher | `google/tipsv2-b14` |
| Towers | **Vision + Text**, cả hai frozen |
| Vision patch size | **14** |
| Teacher preprocessing | RGB `[0,1]`, **không dùng ImageNet normalization**, theo model card |
| Text preprocessing | Text tokenizer/encoder của TIPSv2, không dùng MiniLM tokenizer |
| Teacher patch grid at 224×336 | **16×24 = 384 patches/camera** |
| Teacher embedding dimension | **768** |
| Output supervision | Dense patch–phrase similarity, được hiệu chuẩn/kiểm tra độ tin cậy |

Với phrase `p`:

\[
v^T_i=\operatorname{normalize}(\operatorname{TIPSvision}(I)_i),
\quad
u^T_p=\operatorname{normalize}(\operatorname{TIPStext}(p)).
\]

\[
s^T_{p,i}=\langle v^T_i,u^T_p\rangle.
\]

Những vector phải được lấy ở **đúng common embedding space** theo model interface, không giả định các hidden states bất kỳ đều có thể cosine với text.

Map trên grid 16×24 được chuyển về student grid 14×21 bằng **2D spatial resampling theo image coordinates** sau các bước xử lý ảnh tương ứng. Không dùng reshape tùy ý hay so patch index 1:1.

Phân phối teacher:

\[
P^T_{p,n}=\operatorname{Softmax}_{n\in\mathrm{valid}}
\left(\operatorname{resize}(s^T_p)_n / \tau_T\right),
\]

với `τ_T=0.07` làm khởi đầu; cần đánh giá entropy và positive/negative margin vì cùng một nhiệt độ không bảo đảm calibration trên mọi phrase.

**Không coi `P^T` là ground-truth segmentation.** Teacher có thể gán nhầm gripper thành red block như audit checkpoint cũ.

### 6.2. Teacher B — DINOv3 ViT-L/16 (dense spatial structure)

| Thuộc tính | Quyết định |
|---|---|
| Teacher | Frozen DINOv3 vision ViT-L/16 |
| Vai trò | Dense visual patch representation, local object/region consistency |
| Teacher patch size | **16** |
| Grid at 224×336 | **14×21=294/camera** nếu model/processor cho phép |
| Output | Patch tokens sau khi **loại CLS/register/non-patch tokens** |
| Loss target | Pairwise normalized feature affinities giữa spatial patches |

Cho một camera, gọi teacher dense features là `D ∈ [294,d_T]`:

\[
R^T_{ij}=\langle\operatorname{normalize}(D_i),\operatorname{normalize}(D_j)\rangle.
\]

Nếu teacher input preprocessing hoặc grid khác, resample teacher **spatial features** sang grid student trước khi xây affinity. Không reshape arbitrary transformer tokens có CLS/register.

**Ranh giới nghĩa:** DINOv3 chỉ cung cấp spatial coherence; không biết mục tiêu instruction, không biết target action. Không cộng raw DINO similarity vào semantic confidence mà không qua calibration.

### 6.3. Teacher C — Grounding DINO (language-conditioned region proposals)

| Thuộc tính | Quyết định |
|---|---|
| Teacher | Grounding DINO pretrained, frozen, **offline** |
| Input | RGB + referring phrases theo task |
| Output cơ sở | Box proposals, referring-language scores |
| Candidate cap | **Top 5 boxes / phrase / camera** trước lọc |
| Mục tiêu | Nhãn vùng object/part/surface khi phát hiện đáng tin cậy |
| Tính cần thiết | Có để bổ sung **direct region supervision**, nhưng không nằm trong runtime policy |

Grounding DINO là language-conditioned region proposal teacher. Nó khác với DINOv3; tên gần nhau không đồng nghĩa architecture/training mục tiêu như nhau.

Teacher không đủ sức xác định mọi action-affordance: động tác kéo, xoay, lau mặt bàn hoặc điểm tiếp xúc chính xác có thể không xuất hiện dưới dạng box đơn giản.

### 6.4. Optional offline SAM 2

SAM 2 **không phải một trong ba teacher cốt lõi của kiến trúc student**. Chỉ là tiện ích offline để chuyển validated boxes thành mask candidates và nhất quán theo video.

- Nếu chỉ có trusted bbox, **không** coi toàn bộ rectangle là ground-truth silhouette.
- Nếu có reliable mask, train mask readout bằng mask loss.
- Nếu mask không đáng tin cậy hoặc occlusion mạnh, không phát sinh hard negative cho phần chưa được dán nhãn.
- Không buộc phải dùng SAM 2 để train student; hình thức region supervision được xác định bởi độ tin cậy và kiểu nhãn.

### 6.5. Teacher-cooperative pseudo-label synthesis — thứ tự chuẩn

**Không chọn pipeline TIPS-first hard cropping.** Mọi teacher nhìn full frame; giữ candidate hypotheses chưa kết luận cho đến bước consensus.

1. **Generate candidate regions:** Grounding DINO sinh tập `R={r_j}` với box, phrase và score liên quan; SAM 2 có thể refine mask khi đáng tin cậy.
2. **Estimate semantic agreement:** TIPSv2 đánh giá độ tương ứng của **phrase cụ thể** với từng candidate; dùng relative rank/margin và overlap với patch similarity, không lấy teacher top-1 như ground truth.
3. **Estimate spatial coherence:** DINOv3 giúp đo coherence của candidate, cải thiện mask soft boundary và đánh giá region consistency; không dùng để tự chứng nhận ngữ nghĩa.
4. **Calibrate / abstain:** Các scores được hiệu chuẩn trên một audit set nhỏ có annotation; candidate không đạt độ tin cậy thì `label_valid=0`, không thành background negative.
5. **Construct per-frame targets:** cho mỗi accepted region, lưu label type (`mask` / `box` / `point` nếu được xác thực), phrase, camera id, valid fraction, confidence weight, mapping về image coordinates và episode/frame id.
6. **Make temporally consistent:** Track proposal quality qua frames; nếu nhảy region hoặc biến mất do occlusion, giảm confidence hoặc bỏ frame đó.

Không dùng `M_TIPS` làm binary gate loại visual evidence trong shared student. Student **luôn thấy toàn ảnh hợp lệ**, đặc biệt khi teachers bất đồng.

### 6.6. Cơ chế confidence-aware supervision

Cho annotation ứng viên `j`, gán:

\[
c_j\in[0,1],\qquad m_j\in\{0,1\}.
\]

- `m_j=1` chỉ khi annotation type và region đủ tin cậy.
- `c_j` phản ánh reliability đã kiểm tra/calibrate, không coi một raw Grounding DINO confidence thành true probability.
- Candidate **vắng mặt hoặc bị từ chối** không tạo positive target cũng không tạo background negative trên các vùng không được gắn nhãn.
- Với confidence kém, cho phép sử dụng **teacher similarity loss nhẹ** nhưng không dùng full region localization loss.
- Nếu teacher và manual validation trái nhau, manual validation là reference đáng tin hơn.

### 6.7. Dataset LeRobot không mặc định có những nhãn này

LeRobot format cung cấp theo schema các camera RGB, state/action, metadata tasks, timestamps và episode boundaries. **Boxes, masks, grasp points, 3D poses và camera calibration là optional, không được giả định có sẵn.** Tập `teacher_targets` là **dataset supervision bổ sung offline**, tách biệt nhưng index đồng nhất với RGB của LeRobot.

### 6.8. Validated pseudo-label set

Để có một phép kiểm tra teacher đáng tin trước khi train student:

- **100 frames** được chọn nhiều episode, task, thời gian thao tác và camera; đánh dấu regions thực sự nhìn thấy bằng annotation kiểm tra thủ công.
- Báo cáo teacher precision/recall, pointing accuracy, mask IoU (khi có masks), error cases và tỷ lệ abstain theo từng loại region.
- Mục tiêu **precision ≥ 90% trên accepted pseudo-labels** trong audit set, **không** phải bảo đảm teacher có recall cao hay có khả năng bao phủ mọi task.
- Nếu không đạt độ chính xác tối thiểu, các pseudo-labels chưa đủ điều kiện trở thành localization supervision cứng; điều chỉnh quality gate chứ không buộc student sao chép sai.
- Split audit/validation theo episode, không chia ngẫu nhiên các frames rất gần nhau sang train và validation.

---

## 7. Head 1 — Semantic Alignment Head

### 7.1. Nhiệm vụ đúng

Alignment Head trả lời câu hỏi: **các concept trong instruction tương ứng với evidence nào trong visual tokens?** Nó không tự tuyên bố đâu là grasp point, handle contact point hay 3D destination.

Đầu ra kép:

1. `S`: phrase-conditioned dense semantic logits/maps **[B,6,588]**.
2. `Z_a`: 8 semantic latent tokens **[B,8,384]**, mỗi token đọc toàn cảnh với điều kiện text phù hợp.

### 7.2. Dense semantic map generator

- Visual linear projection: `384→256`.
- Text phrase linear projection: `384→256`.
- L2 normalization theo last dimension.
- Scaled cosine similarity với `τ_sem = 0.07`.
- Mask invalid camera/patch; đánh dấu phrase absent không có loss.

\[
\ell_{p,n}=\frac{\langle
\operatorname{norm}(W_vF_n),
\operatorname{norm}(W_t t_p)
\rangle}{\tau_{sem}}.
\]

\[
P^S_{p,n}=\operatorname{Softmax}_{n\in\mathrm{valid}}(\ell_{p,n}).
\]

Đây là map riêng cho **từng phrase**, không bao giờ dùng `mean_p P^S_p` làm attention bias chung cho tất cả latents.

### 7.3. Tám learnable semantic queries

Khởi tạo tám query embeddings riêng `E_a ∈ R[8,384]`. **Không** chia cứng 4 source + 4 destination vì model hướng tới general manipulation.

Mỗi query chọn task/phrase context bằng content-based weighting:

\[
\alpha_{i,p}=\operatorname{Softmax}_p
\left(\frac{(W_qE_{a,i})^\top W_pt_p}{\sqrt{384}}\right)
\]

chỉ trên valid phrases. Với `P=0`, dùng `t_global` làm text context.

\[
q_{a,i}=E_{a,i}+W_g t_{global}+
W_l\sum_p\alpha_{i,p}t_p.
\]

Do `E_a` khác nhau và `α_{i,p}` khác nhau, query được phép chuyên biệt theo content, nhưng **không có shared spatial attention bias**.

### 7.4. Semantic cross-attention block

| Thành phần | Giá trị |
|---|---|
| Layers | **1** |
| Query tokens | **8** |
| Query/key/value dimension | **384** |
| Internal attention heads | **6**, dim/head 64 |
| K,V | Shared vision `F [B,588,384]` |
| FFN | `384→1536→384`, GELU |
| Norm | Pre-LayerNorm + residual |
| Dropout | `0.0` attention, `0.1` FFN during training |
| Output | `Z_a [B,8,384]` |

\[
Z_a=\operatorname{TransformerCrossBlock}(Q_a,F,F).
\]

Cross-attention softmax chạy trên **valid visual patches** riêng cho từng query. Camera identities và 2D positions đã được giữ trong F.

### 7.5. Hạn chế alignment query collapse

Thiết kế phòng ngừa:

- Query embeddings riêng, phrase associations riêng.
- Không cộng một prior `f(mean phrase map)` giống nhau vào tất cả cross-attention logits.
- Dense semantic maps có supervision từ trusted teacher targets.
- Khi có region targets được duyệt, ít nhất một query có phrase affinity tương ứng được học tập trung vào khu vực đó bằng **weak phrase-query correspondence auxiliary loss** (mục 12). Đây là supervision mềm, không bắt tất cả queries nhìn các vùng khác nhau.
- Không dùng global query-diversity loss lớn, vì nhiều query có thể cần thông tin cùng một object.

### 7.6. Chi tiết về weak query-to-region assignment

Với các phrase có accepted teacher region `R_p`, tính per-query attention mass trong region. Cho phép phrase chọn query phù hợp nhất (soft best-match hoặc permutation-invariant matching), rồi giám sát query được chọn tăng mass trong accepted region.

- Chỉ áp dụng khi `label_valid=1`.
- Không phân vai trước theo index query.
- Một region có thể dùng nhiều queries khi cần.
- Tất cả 8 queries không bắt buộc có object riêng; contextual semantic queries là hợp lệ.
- Không đánh đồng query attention supervision với binary semantic segmentation.

### 7.7. Loss semantic

\[
L_{TIPS}=\frac{
\sum_{b,p} w_{b,p}\,m_{b,p}
D_{KL}(P^T_{b,p}\Vert P^S_{b,p})
}{\sum_{b,p}w_{b,p}m_{b,p}+\epsilon}
\]

với `w_bp` là teacher reliability; `m_bp` là phrase-availability và teacher-validity indicator. KL tính trên cùng valid spatial support/camera và target resize đúng hình học.

Vì TIPS maps có thể sai, **teacher localization reliability quyết định trọng số**; nếu accepted human-checked region mâu thuẫn mạnh với TIPS map, đặt TIPS weight thấp hoặc 0 trên sample đó.

---

## 8. Head 2 — Task-Adaptive Interaction Grounding Head

### 8.1. Nhiệm vụ và output

Đây là **trung tâm của kiến trúc v3**. Grounding Head không phải detector chuyên pick/place mà học một tập nhỏ các interaction-centric regions cần thiết cho instruction tại robot state hiện tại.

Các output:

- Dense spatial feature map `F_g [B,2,294,384]` trước slot pooling.
- **8 competitive interaction slots** `G [B,8,384]`.
- Slot-region logits/maps `M [B,8,2,14,21]`, với valid camera/patch masks.
- Per-slot/per-camera 2D spatial descriptors, region mass và uncertainty.
- Grounded feature tokens `Z_g [B,8,384]` cho action fusion.

Không có output cố định mang tên `source`, `destination`, `grasp`, `place`. Khi task cụ thể cần các vai trò đó, teacher labels và attention có thể tạo tương ứng động.

### 8.2. Dense spatial adapter

Cho mỗi camera độc lập:

```text
F_c [14,21,384]
    -> Conv 3×3, 384→384, padding1 -> GELU -> LayerNorm-equivalent
    -> Conv 3×3, 384→384, padding1 -> GELU -> residual with F_c
    -> F_g^c [14,21,384]
```

- **2 convolution layers**.
- Không trộn `top`/`wrist` như hai phần liền nhau của cùng feature map.
- Camera embedding và 2D patch coordinates giữ cùng hình học với student Vision.
- Adapter receives **soft** semantic evidence từ Alignment dưới dạng một projected per-patch feature/residual, không sử dụng hard gating/hard crop.
- Semantic injection coefficient khởi tạo nhỏ (ví dụ output projection zero-initialized) để student không bị sai lệch quá sớm bởi TIPS wrong maps.

### 8.3. Conditioning bởi language, alignment và robot state

Grounding khởi tạo 8 slots:

\[
G_i^{(0)}=E_{g,i}+
W_t t_{global}+
W_a\operatorname{Pool}(Z_a)+
W_s\operatorname{Pool}(Z_s)
+W_p t_{phrase(i)}.
\]

Trong đó `t_phrase(i)` là soft phrase-mixture riêng của slot `i`, không nhất thiết có role cố định; nếu không có phrase thì dùng `t_global`.

- `E_g,i` là learned slot seed **khác nhau giữa 8 slots**.
- `Z_s` đi vào Grounding **trước slot selection**, không chỉ ở fusion.
- Không truyền vào camera extrinsics giả định khi LeRobot schema không lưu calibration.
- Grounding phải giữ đường gradient từ `L_action` về **slot-to-patch region probabilities**; không detach toàn bộ alignment→grounding→action.

### 8.4. Iterative competitive slot attention

| Thành phần | Giá trị |
|---|---|
| Số interaction slots | **8** |
| Số vòng refinement | **2** |
| Slot width | **384** |
| Competition attention heads | **6**, head width 64 |
| Keys/values | `F_g` sau spatial adapter, camera-aware |
| Slot update | Shared GRU-style update + FFN `384→1536→384` |
| Attention norm bước 1 | **Softmax theo slots** cho từng valid visual patch |
| Attention norm bước 2 | Chuẩn hóa contribution theo patches trong từng slot |
| Patch masks | Có, không cho padding/camera invalid tham gia |
| Output slots | `[B,8,384]` |

Với một attention head, logits:

\[
L_{k,n}=\frac{q_k^\top k_n}{\sqrt{d_h}},\quad d_h=64.
\]

**Competitive assignment:**

\[
P_{k,n}=\frac{\exp(L_{k,n})}{\sum_{k'=1}^{8}\exp(L_{k',n})}.
\]

Với mỗi valid patch `n`, tổng assignment theo 8 slots bằng 1. Không softmax theo patches **trước** competitive normalization.

**Slot aggregation weights:**

\[
\widetilde P_{k,n}=
\frac{P_{k,n}\,m_n}{\sum_{n'}P_{k,n'}m_{n'}+\epsilon}.
\]

\[
u_k=\sum_n\widetilde P_{k,n}\,v_n.
\]

Sau đó update:

\[
G^{(r+1)}=\operatorname{GRUUpdate}(u,G^{(r)})
+\operatorname{FFN}(\mathrm{LN}(G^{(r)})).
\]

Thực hiện **r=0,1** (tổng 2 vòng). Tất cả slots nhìn toàn bộ valid visual tokens; **không hard-crop** theo map student/TIPS.

**Interpretation:** competition giúp tạo inductive bias giảm trùng lặp; không phải bằng chứng toán học rằng từng slot sẽ học object/action-affordance riêng biệt. Multi-head slot competition và masking phải giữ đúng chiều normalized softmax.

### 8.5. Region-mask readout khác attention visualization

Sau hai iterations, ground slot `G_k` và dense spatial tokens `F_g,n` tạo region logits:

\[
\ell^{region}_{k,n}
=\frac{(W_mG_k)^\top(W_f F_{g,n})}{\sqrt{128}},
\]

với `W_m, W_f: 384→128`.

\[
M_{k,n}=\sigma(\ell^{region}_{k,n}).
\]

- Output `M` theo từng slot/camera dạng soft probabilities.
- Đây là **trainable spatial readout**, không mặc định là raw Slot Attention weights.
- Pseudo-mask loss chỉ áp dụng trên reliable matched regions; không có label thì region predictor vẫn nhận action gradient qua soft pooling.
- Các masks được phép chồng lấp (ví dụ surface và tool overlapping camera projection); cạnh tranh nằm ở feature slot assignment, không ép output mask partition cứng toàn ảnh.
- Một background/no-object response có thể tồn tại nội bộ, nhưng không tính thành specialized head thứ tư.

### 8.6. Geometric descriptors tường minh (không cần calibration)

Với mỗi slot `k`, camera `c`, normalize mask weights:

\[
\overline M_{k,c,n}
=\frac{M_{k,c,n}m_{c,n}}
{\sum_{n'}M_{k,c,n'}m_{c,n'}+\epsilon}.
\]

Mean 2D patch coordinates:

\[
\mu_{k,c}=\sum_n\overline M_{k,c,n}(u_n,v_n).
\]

Spatial spread:

\[
\sigma^2_{k,c}=\sum_n\overline M_{k,c,n}
((u_n,v_n)-\mu_{k,c})^2.
\]

Region mass:

\[
a_{k,c}=\frac{\sum_n M_{k,c,n}m_{c,n}}{\sum_n m_{c,n}+\epsilon}.
\]

Visibility/confidence proxy `v_kc`: một scalar được học từ slot/region statistics hoặc chuẩn hóa attention entropy. **Không tự tuyên bố đó là xác suất đúng**; cần calibration nếu dùng cho quyết định khác.

Descriptor per camera (D_geo=6):

\[
g_{k,c}=[\mu_x,\mu_y,\sigma_x,\sigma_y,a_{k,c},v_{k,c}].
\]

Nếu `camera_valid=0`, descriptors/pooled features của camera đó bị mask hoàn toàn.

**Đây là tọa độ 2D ảnh, không phải 3D object/world coordinates.** Không suy ra khoảng cách vật lý EEF–object từ chúng khi không có calibration/depth.

### 8.7. Grounded feature pooling và final `Z_g`

Cho slot `k`, camera `c`:

\[
f_{k,c}=\sum_n \overline M_{k,c,n}\,F_{g,c,n}.
\]

Đưa `f_{k,c}` và `g_{k,c}` qua camera-specific descriptor projection và gộp theo gated learned camera weighting `w_{k,c}`, chỉ trên camera valid:

\[
h_{k,c}=W_{joint}[f_{k,c};g_{k,c};e_c]\in\mathbb R^{384}.
\]

\[
w_{k,c}=\mathrm{MaskedSoftmax}_c(
\mathrm{MLP}([G_k;h_{k,c}])).
\]

\[
Z_{g,k}=\mathrm{LN}\left(G_k+
\sum_{c\in\mathrm{valid}}w_{k,c}h_{k,c}\right).
\]

**Đường `M -> f,g -> Z_g -> Action Decoder` phải differentiable**, kể cả khi `M` chưa có teacher mask label. Điều này khác bản v1 chỉ giám sát dense features trước pooling.

### 8.8. Quan hệ giữa dense feature distillation và region supervision

`F_g` được giám sát qua DINO feature relational loss (mục 12). `M` và `Z_g` được giám sát bằng accepted region pseudo-labels và action loss. Không đánh đồng hai việc này.

### 8.9. Region matching không phụ thuộc slot index

Có thể có `J` teacher region candidates (J biến thiên) và 8 student slots.

- Tạo matching cost gồm localization overlap + phrase semantic compatibility + confidence, có support no-object/unmatched slots.
- Match permutation-invariant (Hungarian hoặc soft optimal transport là các mô hình toán học phù hợp).
- Cho phép tối đa 8 reliable matched regions ở mỗi observation; nếu J>8, ưu tiên top candidates theo task relevance và label reliability, còn lại là unassigned không tạo false negative.
- Không gán slot 0 luôn là block, slot 1 luôn là holder, slot 2 luôn là gripper.
- Với reliable masks: mask loss trên matched slots.
- Với reliable boxes nhưng chưa có masks: supervise **bbox/center/extent** của predicted spatial support, không áp dụng Dice/BCE với nguyên hình chữ nhật như silhouette thật.
- Nếu teacher không detect object thật đang present, không tự biến nó thành ground-truth background.

### 8.10. Multi-camera grounding

- Region maps tồn tại **riêng từng camera**: `[8,2,14,21]`.
- Slot identity là một abstract interaction concept dùng chung across views, **không** mặc định hai maps cùng tọa độ cho cùng một slot ở hai camera.
- Khi camera calibration không có, không áp đặt 3D consistency loss hay triangulation giả.
- State-conditioned cross-camera fusion được học từ demonstration action loss và aligned language.
- Nếu một camera bị occlusion, camera gate có thể giảm trọng số view đó; không chia đều 50/50 bắt buộc.

---

## 9. Head 3 — Robot State Head

### 9.1. Nhiệm vụ

Chuyển `observation.state` tùy robot thành proprioceptive latent representation hữu ích cho **Grounding Head trước spatial selection** và cho **Action Decoder qua Latent Fusion**.

### 9.2. Robot-agnostic state tokenization

Với `D_s` scalar channels:

\[
x_j=\operatorname{NormBySchema}(s_j),
\quad
\ell_j=\mathrm{MLP}_{scalar}(x_j)+e_{channel(j)}.
\]

`D_s` được đọc từ LeRobot metadata, không gán cứng joint angles, EEF pose hoặc gripper semantics nếu schema không có.

- Mỗi numeric channel có identity/type embedding riêng.
- Missing/invalid channel có mask, không đi vào pooling.
- Per-robot state normalization từ training statistics; inference dùng cùng parameters.
- Nếu schema có categorical/discrete gripper signal, encode đúng type và range, không ép nó thành Cartesian EEF pose.

### 9.3. Bốn state queries

| Thành phần | Giá trị |
|---|---|
| State queries | **4 learnable queries** |
| Query width | **384** |
| Layers | **1 cross-attention block** |
| Attention heads | **6**, width64 |
| K,V | State scalar tokens từ `D_s` channels |
| FFN | `384→1536→384` |
| Output | `Z_s [B,4,384]` |

\[
Z_s=\operatorname{StateCrossBlock}(E_s,\ell,\ell).
\]

Không cần state-specific teacher. State Head được học từ action loss, gián tiếp từ grounded task context và normalization constraints.

### 9.4. Early state conditioning

`summary(Z_s)` phải được Grounding Head sử dụng **trước** competitive slots (mục 8.3). Đây là invariant. Nếu state chỉ được concat tại Latent Fusion, implementation không còn đúng bản v3.

---

---

## 10. Recurrent temporal memory và Latent Fusion

### 10.1. Vì sao giữ memory trong một model simple?

Chỉ **4 compact tokens** mang thông tin các observation trước. Không đưa thêm ảnh lịch sử đi qua ViT trong cùng một inference. Bộ nhớ này không phải specialized head thứ tư và không tạo module perception foundation mới.

- State transitions (`approach → contact → continue`) được học từ temporal demonstrations.
- Không cố định các task stages; model không cần nhãn phase.
- Memory được cập nhật bằng recurrent state nhỏ, không có attention history không giới hạn.
- Đối với task một bước, memory vẫn hoạt động nhưng có thể được chú ý ít qua Fusion.

### 10.2. Current tokens

Theo thứ tự bất biến:

\[
Z^{cur}_t=[Z_{a,t};Z_{g,t};Z_{s,t}]
\in\mathbb R^{20\times384}.
\]

Token order luôn gồm 8 semantic + 8 interaction + 4 proprioceptive tokens, với **modality/type embeddings** để Fusion phân biệt xuất xứ.

### 10.3. Memory

\[
M_t=[m_{t-4},m_{t-3},m_{t-2},m_{t-1}]
\in\mathbb R^{4\times384}.
\]

- Start episode: memory gồm zero vectors và **memory_valid=0**; không coi 4 zero memory tokens là observations thực.
- Khi đã thấy `k<4` observations, chỉ `k` tokens thật có validity mask.
- Không bao giờ dùng observation tương lai để tạo memory tại thời điểm `t`.

### 10.4. Temporal fusion block

\[
X_t=[Z^{cur}_t;M_t]\in\mathbb R^{24\times384}.
\]

| Thành phần | Giá trị |
|---|---|
| Fusion depth | **2 Transformer encoder blocks** |
| Hidden dimension | **384** |
| Self-attention heads | **6**, width64 |
| FFN | `384→1536→384` |
| Norm | Pre-LN |
| Dropout | `0.1` FFN, `0.0` attention |
| Mask | Memory validity; 20 current tokens luôn valid |
| Input | **24 tokens** |
| Output | **24 tokens** |

Giữ output tương ứng 20 current tokens:

\[
Z_t=\operatorname{Fusion}(X_t)_{1:20}
\in\mathbb R^{20\times384}.
\]

Không ép action decoder đọc 4 memory tokens thô riêng, vì memory đã được chuyển vào 20 unified tokens.

### 10.5. Recurrent memory update

Tạo summary mới từ chính unified latent sau fusion:

\[
s_t^{mem}=
\mathrm{LN}\left(
W_{mem}[
\mathrm{Mean}(Z_{a,t});
\mathrm{Mean}(Z_{g,t});
\mathrm{Mean}(Z_{s,t});
\mathrm{Mean}(Z_t)
]
\right)\in\mathbb R^{384}.
\]

FIFO update:

\[
M_{t+1}=[M_t[1:4];s_t^{mem}]
\]

trong ngôn ngữ toán học nghĩa là **bỏ token lâu nhất, giữ ba token mới hơn và append summary mới**. Memory index/order phải nhất quán với thời gian; không có quan hệ với query slot indices.

- Cập nhật memory **một lần cho mỗi observation mới**, không phải sau từng predicted action trong chunk.
- Khi policy thực thi 4 actions không quan sát lại, memory không tự giả lập thêm 4 frames chưa thấy.
- Reset khi episode kết thúc hoặc robot bắt đầu task độc lập.
- Không lưu memory giữa hai episodes trong training hay rollout.

### 10.6. Sequence training

- Training sequence length **4 consecutive observations** cùng một episode.
- Sử dụng truncated backpropagation across temporal window; các windows không được nối nhầm episode.
- Không cần label phase/action primitive.
- Nếu frame sampling rate khác control rate, observation timestamps phải được tôn trọng; memory token tương ứng thứ tự quan sát thực.

---

## 11. Direct deterministic Transformer Action Chunk Decoder

### 11.1. Thiết kế chủ ý đơn giản

**Đây không phải diffusion.** Không có Gaussian noise, denoising timestep, DDIM, noise-prediction loss hoặc sampling loop.

Input:

\[
Z_t\in\mathbb R^{B\times20\times384}.
\]

Output:

\[
\widehat A_t\in\mathbb R^{B\times32\times D_a}.
\]

Toàn bộ 32 actions được dự đoán **đồng thời trong một forward pass** từ 32 learnable action-position queries.

### 11.2. Learnable action queries

\[
E_{act}\in\mathbb R^{32\times384}.
\]

Với timestep tương đối `h = 0..31`:

\[
q^{act}_{h}=E^{act}_h+E^{time}_h+W_c\mathrm{Pool}(Z_t).
\]

`E_act` và `E_time` có thể được gộp thành một learned time-indexed query table nếu giữ ý nghĩa tương đương; tại đây đặc tả là learned action queries có vị trí thời gian tường minh.

### 11.3. Direct Transformer decoder

| Thành phần | Giá trị |
|---|---|
| Decoder layers | **6** |
| Width | **384** |
| Internal self-attention heads | **6** |
| Internal cross-attention heads | **6** |
| Head width | **64** |
| Decoder queries | **32** |
| Cross-attention K,V | Unified latent `Z_t [B,20,384]` |
| FFN | `384→1536→384` |
| Normalization | Pre-LN |
| Dropout | `0.1` on FFN/residual in training |
| Autoregressive/causal mask | **Không dùng**; full attention giữa 32 action-position queries |
| Output width | **384** trên mỗi action position |
| Action readout | `384→384→D_a` MLP |

Forward:

\[
H_{act}=\operatorname{Decoder}_6(Q_{act},Z_t,Z_t),
\quad H_{act}\in\mathbb R^{32\times384}.
\]

\[
\widehat A_t=\operatorname{MLP}_{action}(H_{act})
\in\mathbb R^{32\times D_a}.
\]

### 11.4. Action representation contract

`D_a` **phải khớp action feature schema và robot control interface** của dataset.

Có thể là joint positions, joint deltas, Cartesian end-effector target, velocity hoặc hỗn hợp gripper signal; kiến trúc không tự chuyển đổi các loại này một cách hợp lệ nếu thiếu kinematics/metadata.

- Normalize continuous channels theo training stats rồi de-normalize đúng tại output.
- Các channel là discrete/boolean cần có output semantics và loss type phù hợp, không mặc định tất cả là Euclidean regressions.
- Các bounded continuous channels cần range handling phù hợp khi action được đưa cho robot.
- Không có camera/world-frame transformation bắt buộc trong decoder vì dataset không nhất thiết có calibration.

### 11.5. Prediction và execution horizon

**Hai con số khác nhau:**

\[
H_{pred}=32,\qquad H_{exec}=4.
\]

Policy sinh 32 actions một lần; robot **chỉ thực hiện bốn action đầu**, sau đó lấy observation mới, cập nhật memory một lần và dự đoán chunk mới.

- Không thực thi tự động toàn bộ 32-step chunk open-loop.
- Không có adaptive execution horizon ở v3.
- Không có asynchronous generation hay overlap blending ở v3.
- Các giới hạn vận tốc/lực/vùng làm việc và safety interlocks thuộc robot controller và **không được thay thế** bằng heatmap confidence.
- Nếu latency thực tế không đáp ứng được chu kỳ 4 actions, điều đó là **evaluation failure cần báo cáo**, không được âm thầm tăng `H_exec` để tuyên bố latency thành công.

### 11.6. Tại sao chọn 32 dù đã bỏ diffusion?

32 là prediction horizon, không phải số commands buộc phải thực hiện. Nó giúp học consistency của local motion trong demonstrations; decoder parallel-query giữ nguyên số parameter gần như độc lập với horizon. Chi phí tính toán tăng cùng action token count, nhưng kiến trúc vẫn một forward không sampling loop.

### 11.7. Action loss

Với continuous channels:

\[
L_{act}=\frac{
\sum_{b,h,d} m_{b,h}w_d\,
\rho_{\beta}(\widehat A_{b,h,d}-A_{b,h,d})
}{\sum_{b,h,d}m_{b,h}w_d+\epsilon},
\]

trong đó:

- `ρ_β` = Smooth-L1/Huber with `β=0.1` **trên action đã normalize**.
- `m_bh` = validity mask cho future actions trong cùng episode.
- `w_d = 1` khởi đầu cho normalized continuous channels; discrete channels có criterion riêng.
- Mọi future step `h=0..31` có initial weight bằng nhau.

Kèm derivative matching nhẹ:

\[
L_{delta}=\operatorname{MaskedMean}_{h=1..31}
\rho_{0.1}\left([
\widehat A_{h}-\widehat A_{h-1}]-[A_h-A_{h-1}]
\right).
\]

`L_delta` so **chênh lệch dự đoán với chênh lệch thật**; không ép mọi trajectory trơn tới mức làm mất chuyển tiếp cần thiết. Chỉ tính khi hai steps liền kề đều có target hợp lệ.

### 11.8. Kết thúc episode và padding

Nếu remaining actions <32 ở cuối episode, tạo action validity mask; **không** coi padded repeated/fake actions là targets có loss. Không lấy chunk vượt episode boundary.

---

## 12. Training objectives — công thức và trọng số cố định

### 12.1. Tổng quát

\[
\boxed{
L_{total}=\lambda_{act}L_{act}
+\lambda_{delta}L_{delta}
+\lambda_{sem}L_{TIPS}
+\lambda_{spatial}L_{DINO}
+\lambda_{region}L_{region}
+\lambda_{query}L_{query-region}
}
\]

Mỗi loss **masked/averaged theo số lượng positive/valid target elements**, tránh region-dense objective lấn át action chỉ vì có nhiều patches. Các giá trị dưới đây là **hyperparameters đã chốt cho lần triển khai đầu**, không phải hệ số tối ưu được khoa học chứng minh.

### 12.2. DINO patch relational loss

Student dense spatial features sau adapter, per camera:

\[
R^S_{ij}=\langle\mathrm{norm}(F_{g,i}),\mathrm{norm}(F_{g,j})\rangle.
\]

Teacher DINO:

\[
R^T_{ij}=\langle\mathrm{norm}(D_i),\mathrm{norm}(D_j)\rangle.
\]

\[
L_{DINO}=\operatorname{Mean}_{c,i,j:valid}
\left[(R^S_{ij}-R^T_{ij})^2\right].
\]

- Chỉ so within-camera patch pairs, không áp đặt affinity giữa camera frames không hiệu chuẩn.
- Không có CLS/register trong affinity matrix.
- Cùng grid/cùng image preprocessing geometry.
- Mask tất cả invalid patch pairs.
- Không huấn luyện student chỉ để giống raw DINO channel vectors vì d_student và d_teacher khác và bases không cố định.

### 12.3. Region supervision với varying annotation types

Cho tập accepted teacher candidates `G={g_j}` và 8 student region predictions `M={M_k}`, tìm assignment permutation-invariant tối thiểu theo cost hợp lý.

**Mask labels khi có verified masks:**

\[
L_{mask}=0.5L_{BCE}^{pos/neg-valid}+0.5(1-\mathrm{Dice}).
\]

- Negative pixels chỉ đáng tin trong **validated labeled mask**; vùng chưa annotate không trở thành negative mặc định.
- Camera/valid-area mask được áp dụng trước normalization.

**Box-only labels:** supervise predicted mask-derived bounding extent và center bằng robust L1/IoU-compatible box objective trên credible region box; **không** dùng rectangle mask như object silhouette target. Nếu prediction region quá diffuse và box derivative không ổn, supervisory signal có thể là soft center/extent descriptors; mục tiêu vẫn là localization chứ không gán object mask giả.

**Valid point-only labels:** soft point heatmap loss hoặc coordinate loss, chỉ khi điểm đã được đánh giá hợp lệ; không buộc mọi task có grasp points.

Tổng:

\[
L_{region}=\frac{\sum_{matched\ j}c_j L^{type(j)}_j}{
\sum_{matched\ j}c_j+\epsilon}.
\]

Nếu không có valid region trong batch, **`L_region = 0`** cho batch đó, không tạo all-background target.

### 12.4. Alignment query-to-region auxiliary

Khi một phrase có matched reliable region, dùng phrase-to-query assignment để chọn/weight semantic query attention distributions có liên quan, tính loss khuyến khích attention mass trong accepted region:

\[
L_{query-region}=\operatorname{Mean}_{p:\mathrm{valid}}
[-\log(\sum_{n\in R_p} A_{q^*(p),n}+\epsilon)].
\]

- `q*(p)` là best-matched query theo content/attention cost, không phải predetermined slot id.
- Với box-only label, `R_p` là credible rectangle **cho positive mass**, không có nghĩa mọi pixel trong rectangle đều object.
- Không có match đáng tin ⇒ không tính loss này.
- Loss nhỏ để tránh biến semantic queries thành duplicate hard detectors.

### 12.5. Student semantic distillation `L_TIPS`

Như mục 7.7, KL teacher→student trên **cùng grid, cùng phrase và same camera**, teacher reliability weighted. Nếu TIPS map xung đột với verified region, teacher KL được giảm/loại trên pair đó; Grounding DINO không có quyền tự động override annotation thủ công.

### 12.6. Loss weights — Stage 1 vs Stage 2

| Loss | Stage 1 perception | Stage 2 joint policy |
|---|---:|---:|
| `L_action` | **0.0** | **1.0** |
| `L_delta` | **0.0** | **0.10** |
| `L_TIPS` | **0.30** | **0.20** |
| `L_DINO` | **0.10** | **0.05** |
| `L_region` | **1.00** | **0.50** |
| `L_query-region` | **0.10** | **0.05** |

Đây là coefficient *sau khi loss từng loại đã được normalized*. Phải ghi lại raw và weighted loss trong evaluation; nếu gradients một objective lớn hơn quá nhiều, báo cáo scale và impact thay vì âm thầm thay đổi scientific experiment.

### 12.7. Gradient routing

| Source loss | Alignment Head | Grounding Head | State Head | Fusion/Action Decoder | Shared ViT | MiniLM |
|---|---|---|---|---|---|---|
| `L_TIPS` | Có | Không bắt buộc | Không | Không | Có khi unfreeze | Có khi unfreeze |
| `L_DINO` | Không bắt buộc | **Spatial adapter** | Không | Không | Có khi unfreeze | Không |
| `L_region` | Có thể qua semantic conditioning | **Có: slots, masks, pooler** | Có thể qua state conditioning | Không trực tiếp | Có khi unfreeze | Có khi unfreeze |
| `L_action` | **Có** | **Có** | **Có** | **Có** | Có khi unfreeze | Có khi unfreeze |

- Nếu cần ổn định teacher losses, **detach các nhánh teacher target**, không detach action gradient của Grounding mask→pooling.
- Không dùng DINO loss để ép update MiniLM.
- Không dùng KL semantic teacher làm surrogate cho region mask loss.
- Conflict trên shared ViT vẫn có thể tồn tại; tách specialized heads giảm cạnh tranh ở tham số riêng chứ không loại bỏ conflict toàn bộ.

---

## 13. Training từ một model policy mới hoàn toàn

### 13.1. Khởi tạo mới vs pretrained

**Pretrained initialization được phép:** ViT-S/16 vision và MiniLM-L12-H384 text.  
**Bắt buộc initialize mới:** Alignment Head, Grounding Head, State Head, memory, Fusion, action query embeddings và Decoder.  
**Không được load:** TIPS-Ground-VLA 20k student weights, shared-prior bias module, old action decoder, old query embeddings.

Teacher checkpoints luôn frozen, không được tính vào student parameter budget.

### 13.2. Stage 0 — target quality specification

Trước stage perception, teacher pseudo-regions phải tồn tại ở format confidence/validity rõ ràng, hoặc đánh dấu sample không có region labels. Dataset gốc vẫn được giữ nguyên LeRobot schema.

Không xem việc sinh pseudo-label hàng loạt là bằng chứng grounding đã đúng. Một teacher audit nhỏ theo episode/task/camera là điều kiện khoa học cần thiết.

### 13.3. Stage 1 — Perception pretraining

Mục tiêu: huấn luyện dense semantic alignment và task-adaptive grounded region representation **trước khi action decoder có thể tìm shortcut**.

Modules chủ động:

- Vision ViT-S/16: pretrained; 8 blocks đầu frozen, 4 blocks cuối có LR thấp.
- MiniLM: frozen trong Stage 1.
- Alignment Head: train.
- Grounding Head: train.
- State Head: train theo weak context/region loss **chỉ khi region targets có thể sử dụng state**; nếu không, State Head được init và chủ yếu học tại Stage 2.
- Fusion/Decoder: chưa optimize trong Stage 1.

Loss = `0.30 L_TIPS + 0.10 L_DINO + 1.00 L_region + 0.10 L_query-region`.

Thời lượng mặc định: **20 epochs**; model được chọn theo held-out region/semantic localization metrics chứ không chỉ training loss. Nếu label coverage rất thấp, không được kết luận stage thành công chỉ vì loss nhỏ.

### 13.4. Stage 2 — Grounded action training

Modules chủ động:

- Train 3 specialized heads + fusion + direct decoder + recurrent memory.
- ViT 8 blocks đầu frozen; 4 blocks cuối fine-tune LR nhỏ.
- MiniLM 10 blocks đầu frozen; 2 blocks cuối fine-tune LR rất nhỏ.
- Semantic/region/dense spatial losses tiếp tục làm auxiliary objectives.
- Training sequence dài 4 observations; mỗi observation có 32-step future action target và masking đúng episode boundary.

Loss = `L_action + 0.1 L_delta + 0.2 L_TIPS + 0.05 L_DINO + 0.5 L_region + 0.05 L_query-region`.

Thời lượng mặc định: **80 epochs**, early selection theo held-out action error **và** localization/robot rollout metrics; không chọn checkpoint chỉ bằng training action loss.

### 13.5. Không teacher forcing hoàn toàn ở grounded features

Khi Stage 2 action decoder nhận `Z_g`, nguồn **chính luôn là region predictions của student**. Teacher masks là supervision/regularization, không phải một input inference bắt buộc vào decoder. Điều này tránh train/deploy distribution gap khi predicted masks không hoàn hảo.

### 13.6. Hyperparameters training đã chốt

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Optimizer betas | `(0.9, 0.95)` |
| Weight decay | `0.05` (không áp lên bias/norm/position embeddings) |
| Head/Fusion/Decoder LR | `1e-4` |
| ViT last-4-block LR | Stage 1: `1e-5`; Stage 2: `5e-6` |
| MiniLM last-2-block LR | Stage 1: frozen; Stage 2: `2e-6` |
| Batch size (temporal windows) | `8` |
| Gradient accumulation | `4` (effective 32 temporal windows) |
| Temporal window length | `4` observations |
| Stage 1 epochs | `20` |
| Stage 2 epochs | `80` |
| LR scheduler | Cosine decay, 5% warmup steps mỗi stage |
| Gradient clipping | Global norm `1.0` |
| LayerNorm eps | `1e-5` |
| FFN activation | GELU |
| FFN expansion | 4× |
| Attention dropout | `0.0` |
| FFN/residual dropout | `0.1` training; inactive inference |
| Action Huber β | `0.1` trên normalized action |
| Action horizon | `32` |
| Execution horizon | `4` |
| TIPS student tau | `0.07` |
| Teacher candidate max | 5/phrase/camera |
| Teacher phrase cap | 6/instruction |

Đây là một **cấu hình single-track** nhằm dễ kiểm chứng, không phải bảng các phương án thử song song. Nếu thực nghiệm dẫn tới thay đổi, mỗi thay đổi được coi là một phiên bản/ablation mới, không được trộn vào thông số v3 ban đầu.

### 13.7. Dữ liệu và validation split

- Split theo **episodes**, tốt nhất thêm held-out initial object poses/viewpoints/tasks nếu data có.
- Không để frames gần nhau cùng episode nằm cả train và validation.
- Task texts được lấy từ LeRobot metadata phù hợp, không hard-code ba câu pick/place.
- Spatial pseudo labels chỉ dùng cho quan sát/region thật có thể kiểm tra; không dùng unrelated test-task pseudo labels làm supervision sai.
- Mọi normalizations được fit trên training split, không leak validation/test statistics.

---

## 14. Inference efficiency và cache — chỉ các tối ưu thực sự hợp lệ

### 14.1. Một inference step

1. Nhận observation hai camera hiện tại, robot state và task text.
2. Dùng cached MiniLM text nếu instruction không đổi.
3. ViT-S/16 xử lý **hai camera qua shared weights**.
4. Alignment Head tạo dense maps và tám semantic tokens.
5. State Head tạo bốn proprioceptive tokens.
6. Grounding Head xử lý hai-round competitive attention và grounded region pooling.
7. Temporal Fusion hợp nhất current 20 tokens với 4 memory tokens.
8. Direct decoder **một lần** tạo `[32,D_a]` actions.
9. Memory update; thực thi **4** actions rồi observe lại.

### 14.2. Cache có ích

**Text cache**: encoder outputs/tokenization cố định khi instruction không đổi. Invalidate khi instruction hoặc text-side conditioning thực sự thay đổi.

**Temporal summary cache**: `M_t [4,384]` giữ summaries của các observations thật trước đó. Reset theo episode; bộ nhớ có hạn.

**Intermediate K/V trong một lần forward**: có thể tránh tính lặp lại nếu cùng tensor cố định được sử dụng nhiều lần trong một module, nhưng không làm đây thành yêu cầu runtime độc lập.

### 14.3. Không cache thứ không có lợi/không hợp lệ

- **Không có diffusion** ⇒ không có conditioning KV cache giữa denoising steps.
- Action decoder dự đoán song song 32 positions **một lần**, không autoregressive ⇒ không có GPT-style growing self-attention KV cache.
- Ảnh camera khác nhau mỗi observation ⇒ không reuse stale visual K/V giữa frames như data hiện tại.
- Grounding queries thay đổi theo state/frame ⇒ không cache slot assignments xuyên observations.
- Không cache teacher embeddings vào inference graph; teacher không deploy.

### 14.4. Efficient attention

- ViT self-attention: được phép dùng fused memory-efficient exact attention kernels tương đương FlashAttention, tùy backend/hardware.
- Semantic cross-attention: `Q_a=8` với `N≈588`; attention matrix tương đối nhỏ, không cần Deformable Attention ngay từ v3.
- Competitive Grounding: chỉ **2 rounds × 8 slots × 588 visual tokens**, với key/value projections có thể reuse trong cùng observation vì F_g không đổi qua 2 rounds.
- Fusion: 24 tokens; Decoder: 32 action queries × 20 fused latents; ưu tiên giữ toán tử đơn giản, tương thích nhiều thiết bị.
- GQA không phải bắt buộc vì chiều dài K/V ngắn; không thêm phức tạp chỉ để có tên một kỹ thuật tối ưu.

### 14.5. Số parameters và latency là hai metric khác nhau

Một model ~85M có thể vẫn chậm do **2× visual passes**, image resolution, runtime overhead và preprocessing. Teacher weights không được tính vào inference parameters. Cần đo latency model student thực tế theo batch1 và device deploy, gồm image ingest→action output, nhưng *không mặc định kết luận đáp ứng realtime chỉ vì dưới 100M*.

---

## 15. Parameter budget dự kiến

Đây là **ước lượng** cho kiến trúc đã khóa `d=384`, không phải kết quả đếm parameter đã thực thi. Với các chi tiết triển khai như normalization, value projection và embeddings, tổng có thể sai lệch; cần báo cáo measured exact count sau khi dựng model.

| Module | Thiết kế | Ước tính |
|---|---|---:|
| Shared pretrained ViT-S/16 | 12 blocks, 384-width | ~22M |
| MiniLM-L12-H384 | 12 blocks, 384-width | ~33M |
| Alignment Head | Dense projections + 1 cross-attn/FFN + phrase-conditioning | ~2.5–3.0M |
| Grounding Head | 2× Conv 3×3, 2 rounds competitive slots, readouts/pooling | ~4.5–6.0M |
| Robot State Head | Channel tokenizer + 1 cross-attn/FFN | ~2.0–2.5M |
| Temporal Fusion | 2 transformer blocks, d384 | ~3.5–3.7M |
| Memory / embeddings / adapters | 4 memory, camera/task/type embeddings, projections | ~1–2M |
| Action Decoder | 6 layers × self/cross-attn+FFN, d384 | ~14–15M |
| Action readout / misc | `384→384→D_a` | ~0.2–1M |
| **Total preliminary** | **Student inference model only** | **~80–88M** |

**Budget rule:** giữ dưới ~100M thay vì thêm tham số vô ích để đạt chính xác 100M. Muốn nâng hidden width/depth để sát 100M sẽ là ablation **sau** khi các giả thuyết grounding đã được kiểm chứng.

### 15.1. Bộ phận không tính vào budget

- TIPSv2-B/14 (~86M vision + ~110M text theo model card).
- DINOv3 ViT-L/16 teacher.
- Grounding DINO teacher.
- SAM 2 optional offline.
- Teacher caches và offline annotations.

---

---

## 16. Kiểm chứng kiến trúc: các chỉ báo bắt buộc phải tồn tại

**Một model chạy được forward, giảm training loss hoặc sinh ra heatmaps đẹp chưa đủ chứng minh kiến trúc làm việc đúng.** Các kết quả audit của v1 cho thấy chính điều đó.

### 16.1. Invariant shape và modality

| Kiểm tra | Kỳ vọng |
|---|---|
| Top + wrist valid | `F [B,588,384]`; đúng 294 patches/view |
| Top only | Wrist patches hoàn toàn masked; không đóng góp cho loss/readout |
| Wrist only | Top patches masked tương tự |
| Text unchanged | Text embedding cache không thay đổi prediction ngoài sai số số học |
| Text changed | Cache invalidated; text conditioning được tính lại |
| State missing channel | Channel không hợp lệ bị masked, không thay bằng đặc trưng EEF giả |
| No teacher region | Region auxiliary loss bằng 0; action learning vẫn xác định |
| Phrase-less task | Không phát sinh NaN / padded phrase attention; global instruction conditioning còn hoạt động |
| Episode restart | `M_0` reset, memory validity zero |
| Last N frames episode | Future action validity masks chính xác; không leak sang episode kế tiếp |
| Output | `[B,32,D_a]`, một forward, không sampling loop |
| Student inference | Không import/depend teacher runtime, không DINOv3, TIPSv2, Grounding DINO, SAM 2 |

### 16.2. Teacher quality metrics

Báo cáo trên held-out manually verified frames:

- `TIPS teacher pointing accuracy` theo từng referring phrase/object/part và camera.
- `Grounding DINO accepted region precision`, recall, false negative rate, calibration và abstention rate.
- `DINO feature within-region vs between-region coherence` trên accepted annotations, chỉ làm metric structure.
- `Teacher disagreement rate` và thống kê các trường hợp `TIPS đúng/GDINO sai`, `GDINO đúng/TIPS sai`, cả hai sai.
- Tỷ lệ training frames có reliable mask, bbox-only, no region supervision.

Phân biệt **precision của accepted labels** và **coverage**; một bộ lọc chỉ nhận 2% frames có thể rất chính xác nhưng tạo supervision không đủ cho policy.

### 16.3. Student Alignment metrics

- `Student phrase pointing accuracy` trên verified object/part region.
- Top-k localization hit rate, IoU/mass-in-region khi type label hỗ trợ.
- `TIPS→student semantic KL` **chỉ trên teacher-targets đáng tin**.
- So sánh student và teacher trên cùng ảnh, cùng phrase, đúng spatial warp.
- Text counterfactual trên **một ảnh có cả hai object/regions hiện diện**; không đánh giá model sai khi instruction yêu cầu tìm object không xuất hiện.

### 16.4. Competitive Slot/Grounding metrics

- Raw slot assignment tensor `[B,6 heads,8 slots,588 patches]` trong mỗi iteration, không average qua 8 slots khi audit.
- Pairwise slot-attention cosine similarity và slot feature cosine similarity across valid patches.
- Slot utilization/occupancy, số active slots, entropy của competitive assignment.
- Matched region precision, mask IoU trên mask-GT, box IoU trên bbox-GT và point accuracy khi có point-GT.
- Per-camera region accuracy và confidence calibration.
- Kiểm tra 8 slots có còn collapse vào một location trên phần lớn tasks và observations không.

**Không đặt mục tiêu cứng `all 8 slots must disagree`.** Một vùng lớn, một task đơn giản hoặc một view bị che có thể khiến một số slots cùng thu thập contextual evidence. Tiêu chí là task-appropriate localization và action success.

### 16.5. Student action metrics

- Validation normalized SmoothL1/L1 trên first 4 steps và full 32 steps, tách riêng.
- Action derivative error so với demonstrations.
- Compare action responses khi chỉ thay ảnh, chỉ thay instruction hoặc chỉ thay state trong những counterfactual **hợp lệ về vật lý**.
- Action error cùng và không cùng 4-frame temporal history để xác định temporal memory có tác dụng hay không.
- Episode/robot rollout success rate, completion time, error types và số lần robot tiến nhầm vùng.
- End-to-end latency (camera observations → predicted 32-step chunk), memory usage, parameter count và update frequency trên đúng hardware mục tiêu.

### 16.6. Grounding thật sự điều kiện hóa action?

Một kiểm tra nhân quả bổ sung:

- Với perception/student giữ nguyên, can thiệp có kiểm soát vào **predicted region features** trước Fusion, không chỉ thay màu heatmap hiển thị.
- So sánh action error khi grounded features đúng với khi chúng bị ablate, shuffle hoặc replaced by verified oracle regions.
- Nếu action không đổi có ý nghĩa khi region features bị xóa, model có thể đang dựa chủ yếu vào state/instruction trajectory shortcuts.
- Oracle region test chỉ có ý nghĩa nếu oracle region được đưa qua **cùng pooling/coordinate readout**, không phải vẽ region đúng trên giao diện.

Không suy ra causal grounding chỉ từ raw attention visualization.

---

## 17. Failure modes từ checkpoint v1 và giải pháp v3 có thể kiểm chứng

| Vấn đề quan sát v1 | Giải pháp kiến trúc v3 | Kiểm chứng đúng |
|---|---|---|
| Tám alignment queries nhận cùng bias, collapse | Query-specific phrase conditioning; **không shared spatial bias**; weak region-query alignment | Raw query maps khác nhau theo semantic evidence khi task cần |
| Tám grounding queries cùng hotspot | Competitive interaction slots + explicit region supervision | Slot utilization, region IoU, action reliance |
| TIPS `red block` đôi khi nhìn gripper | Confidence-gated TIPS distillation; verified region proposals | Teacher reliability per phrase; student region accuracy |
| Student học sai ngay cả khi TIPS đúng | Same-frame/grid target alignment, per-phrase KL valid masks | Teacher/student pointing and KL curves |
| DINO spatial loss không chỉ tới target | Giữ DINO như **dense auxiliary**, không gọi đó là localization GT | Region localization metrics độc lập |
| Grounding latent không giữ coordinates | Soft region masks + weighted 2D centroids/spreads/area | Geometric descriptor localization error |
| Source/destination bị average | **Không gộp phrase maps**; variable task phrases/regions, role-agnostic slots | Task-conditioned multi-region matching |
| State quá muộn | State Head conditions Grounding initialization | Compare object/state counterfactual response |
| Single-frame temporal ambiguity | 4-step recurrent latent memory | Controlled pre/post-contact sequence tests |
| Robot theo attention sai region | Verified pseudo-region supervision + differentiable grounded readouts | Oracle region test, robot rollout success |
| 12-step deterministic chunk | Direct 32-step transformer chunk, execute 4 | Horizon-conditional action errors + latency |

Bảng là **design-to-failure mapping**, chưa phải minh chứng hiệu quả được xác nhận thực nghiệm.

---

## 18. General manipulation contract

Grounding Head phải hỗ trợ **các loại vùng và quan hệ tương tác**, không chỉ object center.

| Ví dụ task | Spatial entities mà model có thể cần | Thông tin không được hard-code |
|---|---|---|
| Pick/place | Object, gripper, destination receptacle | Grasp/placement phase |
| Stack | Objects, support face, relative arrangement | Block order/pose |
| Open drawer | Handle, drawer front, robot gripper | Pull direction và contact state |
| Press button | Button/actuator, tool/finger | Contact precision, press direction |
| Push object | Contact surface, manipulated object, obstacles | Push trajectory |
| Rotate object | Object, orientation cues, contact region | Rotation direction |
| Wipe surface | Target surface, tool, unvisited/visited areas | Coverage progression |
| Pour | Source container, target, container orientation | Tilt dynamics, fill state |
| Insert | Hole/peg, alignment edges, EEF | Pose accuracy, insertion axis |
| Fold cloth | Cloth corners/edges, hands/tools | Deformable geometry |

**Grounding DINO masks chỉ cho biết entity localization**; nó không cung cấp đáp án vật lý chính xác về direction, torque, force, contact mechanics hoặc soft-body state. Robot demonstrations và temporal observations phải học các phần còn lại. Kiến trúc không bảo đảm học tốt task không có trong training data.

### 18.1. Dynamic semantic set thay vì source/destination hard-coding

Mỗi instruction có tập dynamic language concepts `C={c_1,...,c_P}`. Các `c_p` có thể là object/part/surface/tool; task-level contextual tokens mang động từ, hướng và quan hệ.

- TIPS cho per-concept semantic maps.
- Grounding DINO/SAM 2 sinh candidate regions cho những concepts có thể localize.
- DINOv3 dạy spatial coherence quanh mọi visual regions, kể cả unlabeled context.
- Eight Grounding slots học interaction-relevance từ action trajectories, không bắt buộc số slots = số concepts.
- Không cần `source` và `destination` luôn có mặt; khi chúng xuất hiện, model có thể học qua phrase/region matching.

### 18.2. Giới hạn về geometry

Không có camera calibration/depth thì mọi 2D coordinates chỉ là **image-relative priors**. Các thao tác thay đổi góc camera lớn có thể vẫn khó. `camera_id_embedding` không bằng 3D calibration. Nếu một phiên bản tương lai bổ sung camera intrinsics/extrinsics hoặc depth, đó là extension ngoài baseline v3, không phải giả định trong spec này.

---

## 19. Quy tắc kiểm tra tính nhất quán của hyperparameters

Mỗi implementation được coi là đúng v3 khi thỏa đồng thời:

```text
Student backbone: ViT-S/16     12 layers, d384, 6 heads
Student text: MiniLM-L12-H384  12 layers, d384
RGB: two cameras max, 224×336; patch tokens per valid view = 294
Alignment: 8 queries, 1 cross-attention block, 6 internal heads
Grounding: 2 conv spatial layers, 8 competitive slots, 2 update rounds, 6 internal heads
State: 4 queries, 1 cross-attention block, 6 internal heads
Current tokens = 8+8+4 = 20
Memory = 4 tokens
Fusion input = 20+4 = 24 tokens
Fusion = 2 Transformer blocks, d384
Unified latent = 20×384
Action decoder = 6 Transformer decoder blocks, d384, 6 heads
Action queries = 32, output = 32×D_action
Training teachers = TIPSv2-B/14 V+T + DINOv3-L/16 + Grounding DINO
Inference teachers = none
Diffusion = none
Prediction horizon = 32
Execution horizon = 4 (fixed)
Student target parameter range = 80–95M, near 100M upper budget
```

Nếu số lượng cameras thật chỉ có 1, masks phải bảo đảm slot readout và temporal fusion hợp lệ. Nếu `D_s`/`D_a` khác nhau giữa robots, module input/output theo schema tương ứng; không ép chung chiều state/action một cách tùy tiện.

### 19.1. Các chiều **không** được hoán đổi

- `Alignment 8 queries` **không** phải số internal attention heads (=6).
- `Grounding 8 slots` **không** phải 8 object categories hoặc 8 fixed task stages.
- `DINOv3` không phải Grounding DINO, dù tên đều chứa DINO.
- `Action prediction 32` **không** có nghĩa execute 32.
- `Memory 4 tokens` **không** phải 4 extra image encodings từ ViT mỗi lần inference.
- `TIPS image patches 384/view` **không** phải `student image patches 294/view`.
- `Student d=384` **không** phải teacher TIPS embedding dim 768.
- `Spatial region logits` **không** phải normalized query-attention map theo định nghĩa.

---

## 20. Tiêu chí hoàn thành ở mức nghiên cứu và bàn giao kiến trúc

### 20.1. Architecture correctness gates

1. **Structural:** có đúng 3 heads và module/tensor shapes như mục 19.
2. **Data correctness:** cameras, patch grid, padding, phrase validity, time windows và action masks hoạt động nhất quán.
3. **Teacher separation:** cả ba teachers chạy training/offline, không có dependency khi deploy.
4. **Alignment correctness:** student dense maps được giám sát từ TIPSv2 both towers, có reliability gate và mapping patch grid đúng.
5. **Grounding correctness:** DINO spatial relation được giám sát **trước slots**; region readout được giám sát **sau slots** và có action gradient xuyên soft region pooling.
6. **Non-collapse:** báo cáo raw per-query/per-slot attention đúng shape và metrics, không chỉ averages/rendered atlas.
7. **Robot-state conditioning:** `Z_s` có đường vào Grounding Head **và** Fusion.
8. **Temporal causality:** memory không leak future và reset theo episode.
9. **Action correctness:** deterministic 32×D chunk, no diffusion, first 4 executed, validity masked.
10. **Efficiency:** đo true inference parameter count, latency và bộ nhớ student; không tính teacher vào parameter target nhưng có ghi training cost.

### 20.2. Performance evidence gates

Không thể đặt một robot success rate tuyệt đối hợp lý trước khi biết dataset/task difficulty. Dù vậy, **cần có ba phép so sánh để bảo vệ giả thuyết**:

- **Teacher/trusted-region audit:** Grounding DINO+TIPS+DINO tạo accepted labels tốt hơn so với chỉ dùng raw TIPS maps ở những trường hợp TIPS nhầm gripper/object.
- **Perception-grounding eval:** Model mới có pointing/region accuracy tốt hơn baseline v1 trên episode/test images chưa dùng train, không đánh đổi bằng teacher label leakage.
- **Robot rollout:** Policy mới thành công nhiều hơn và ít hướng robot vào background hơn trên test tasks; nếu perception tốt nhưng rollout không tốt, cần nghiên cứu action/data/control chứ không coi grounding metrics là proxy tuyệt đối.

### 20.3. Những kết luận **không** được rút ra chỉ từ spec

- Không kết luận teacher consensus chắc chắn đúng.
- Không kết luận competitive attention chắc chắn loại collapse.
- Không kết luận ~85M parameters bảo đảm edge realtime.
- Không kết luận 32-step deterministic regression xử lý mọi multimodal trajectory.
- Không kết luận vùng attention của một query là affordance target.
- Không kết luận general architecture = unseen-task generalization.

---

## 21. Tóm tắt mục tiêu cho coding agent (không phải implementation guide)

**Xây một policy TIPS-Ground-VLA v3 hoàn toàn mới** có 1 shared ViT-S/16, MiniLM-L12-H384, đúng ba specialized heads, task-adaptive interaction slots và một direct 32-step action decoder. Trong training, học semantic TIPSv2 từ cả vision+text, dense spatial DINOv3 và trusted Grounding DINO region proposals; không biến teacher similarities thành hard masks. Student phải dùng language+state để chọn grounded regions; mỗi region cung cấp visual feature + 2D geometric descriptors; action loss phải cập nhật được Grounding Head. Temporal context chỉ gồm 4 cached summary tokens. Model phải chạy inference không cần bất kỳ teacher nào và không diffusion. Không dùng checkpoint cũ làm khởi tạo policy.

---

## 22. Nguồn chính thức và papers để tham khảo

Các tài liệu sau dùng để định hướng **các thành phần**, không có paper nào được tuyên bố mô tả chính xác toàn bộ proposed TIPS-Ground-VLA v3:

### Vision–language semantic teacher

**[R1] TIPSv2 — official pretrained model card**  
https://huggingface.co/google/tipsv2-b14  
Thông số B/14, text/image towers, patch-level outputs, input `[0,1]` **không ImageNet normalization**, image patch size 14, native text embedding.

**[R2] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment (CVPR 2026)**  
https://arxiv.org/abs/2604.12012  
Mục tiêu pretraining patch–text correspondence; không phải action-affordance ground truth.

### Dense visual teacher

**[R3] DINOv3 — Meta official publication**  
https://ai.meta.com/research/publications/dinov3/  
Dense visual features, Gram anchoring và self-supervised spatial representation.

**[R4] DINOv3 technical report**  
https://arxiv.org/abs/2508.10104

### Language-conditioned region proposals

**[R5] Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**  
https://arxiv.org/abs/2303.05499  
Text-conditioned object boxes / referring expressions, không phải DINOv3 representation distillation.

**[R6] Grounded SAM 2 — reference project**  
https://github.com/IDEA-Research/Grounded-SAM-2  
Offline box-to-mask / video tracking workflow; optional, never on the robot inference path.

### Attention and action chunking

**[R7] Object-Centric Learning with Slot Attention**  
https://arxiv.org/abs/2006.15055  
Competitive attention over slots, iterative updates; tham khảo cho Grounding Head, **không** bảo đảm grounded affordances.

**[R8] FlashAttention-2**  
https://arxiv.org/abs/2307.08691  
Efficient exact attention kernels; tính toán nhanh không tự giải quyết query collapse.

**[R9] ACT — LeRobot model overview**  
https://huggingface.co/docs/lerobot/act  
Action chunking transformer làm reference khái niệm, **không** reuse checkpoint hoặc architecture training assumptions đặc thù của TurboVLA.

### Dữ liệu / pretrained student backbones

**[R10] LeRobotDataset v3 format**  
https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx  
Schema `meta/info.json`, `meta/stats.json`, `meta/tasks.jsonl`, tabular states/actions, multi-camera videos, episode metadata và `delta_timestamps`. Không mặc định bbox/mask/calibration.

**[R11] MiniLM-L12-H384 official model card**  
https://huggingface.co/microsoft/MiniLM-L12-H384-uncased  
12 layers, 384 hidden size, ~33M params, pretrained weights cần downstream adaptation.

**[R12] DeiT / ViT-S/16 reference**  
https://arxiv.org/abs/2012.12877  
Pretrained small vision transformer; reference for 12 layers, 384 width, 6 heads, patch16 variant.

---

## 23. Cuối cùng: một sơ đồ duy nhất chốt toàn bộ kiến trúc

```text
         FROZEN TEACHERS — TRAINING / OFFLINE ONLY

    TIPSv2 Vision + Text      DINOv3 Vision      Grounding DINO
           |                      |                     |
     patch-text maps        dense patch affinity   region boxes
           |                      |                     |
           +----------------------+---------------------+
                                  |
                       verified/confidence-aware
                        targets + ignore masks
                                  |
                 perception/grounding auxiliary losses
                                  |
                                  v
 ====================================================================
              STUDENT TIPS-GROUND-VLA v3 (NEW POLICY)
 ====================================================================

 I_top [224,336]         I_wrist [224,336]       Task instruction
          |                     |                      |
          +----------+----------+                      |
                     v                                 v
           Shared ViT-S/16                         MiniLM-L12
           12 blocks, d384                     12 blocks, d384
                     |                                 |
                     v                                 v
            F [588,384]                         T [64,384]
                     |                                 |
                     +-----------------+---------------+
                                       |
                                       v
                               ALIGNMENT HEAD
                        8 language-conditioned queries
                          1 cross-attn block, 6 heads
                              phrase-dense maps
                                       |
                                  Z_a [8,384]
                                       |
                                       +--------------------+
                                                            |
      robot state [D_s]                                  GROUNDING HEAD
              |                                             ^
              v                                             |
         STATE HEAD --- Z_s [4,384] -------------------------+
        4 queries, 1 block                                  |
                                                             |
                                            2 conv spatial blocks
                                          8 competitive interaction
                                          slots, 2 update iterations
                                                   |          |
                                         region maps     2D descriptors
                                                   |          |
                                                   +----+-----+
                                                        v
                                                  Z_g [8,384]
                                                        |
                            Z_a [8,384]   Z_g [8,384]   Z_s [4,384]
                                 +-------------+-------------+
                                               |
                                         current [20,384]
                                               |
                        cached history [4,384]-+
                                               |
                                               v
                                      TEMPORAL FUSION
                                   2 blocks, d384, 6 heads
                                               |
                                         Z_t [20,384]
                                               |
                              +----------------+----------------+
                              |                                 |
                        update memory                     ACTION DECODER
                         [4,384]                  6 blocks, d384, 6 heads
                                                         32 queries
                                                              |
                                                       [32, D_action]
                                                              |
                                               EXECUTE FIRST 4 ACTIONS
                                                              |
                                                    NEXT OBSERVATION

  NO TURBOVLA CHECKPOINT / NO DIFFUSION / NO TEACHERS AT INFERENCE
                     STUDENT TARGET: ~80–95M PARAMETERS
```

**End of architecture specification — TIPS-Ground-VLA v3.0.**
