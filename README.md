# MoDANet — dựng lại bằng PyTorch

Doan et al., *"MoDANet: Multi-Task Deep Network for Joint Automatic Modulation Classification
and Direction of Arrival Estimation"*, IEEE Comm. Letters 26(2), 2022. DOI 10.1109/LCOMM.2021.3132018

Nguồn chân lý: `../MODANET_INSTRUCTION.md` → `configs/model_config.py`. **Khi instruction và bài báo khác nhau, ƯU TIÊN instruction** (xác nhận 2026-09-22). Các chỗ khác bài được ghi lại ở §3 để tham khảo.

---

## 0. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Model (Type A/B/C, k ∈ {3,5,7,9}, 4 preset) | ✅ **457,413 tham số** (Type A, k=7, conv output 1×1), khớp mọi giá trị vàng của instruction |
| Test (`pytest tests/`) | ✅ 196 passed (có test CPU≡GPU, pickle/spawn/fork, tự chọn batch/worker). Đã chạy với NumPy 1.26 và 2.2 |
| Chạy trên mọi máy | ✅ CUDA mọi đời GPU / Apple MPS / CPU; Linux / Windows / macOS; resume được khi đổi máy. Xem §6 (vast.ai; server Arch Linux: §6.6, Debian: §6.8) |
| `scripts/summary.py` | ✅ khớp 100% bảng PHẦN 2.4 + 2.5 + 6B.2 |
| GPU | ✅ GTX 1050 2 GB, torch 2.6.0+cu124 (sm_60 tương thích sm_61), kernel test OK |
| Dataset đầy đủ | ✅ 450,120 file, không thiếu, mỗi tổ hợp đúng 10 file, nhãn tên file ≡ tên thư mục |
| Convert memmap fp16 | ✅ `data_processed/x_float16.npy` (9.22 GB, 20 phút 48 giây trên HDD). Round-trip fp16: sai số tuyệt đối ≤ 3.9e-3, tương đối ≤ 4.9e-4 |
| Chia tập | ✅ `data_processed/splits/random_seed42.npz` (tạo một lần, không ghi đè) |
| Train / inference | ✅ Type A/B/C train đầy đủ trên GTX 1050 (`outputs/typeA,B,C`), early-stopped (patience 8): A epoch 24 (best 16), B epoch 18 (best 10), C epoch 18 (best 10) |
| `scripts/evaluate.py` | ✅ AMC accuracy + DOA RMSE theo SNR (31 mức) cho Type A/B/C, xuất `outputs/eval/per_snr.csv` + Fig. 3(b) — xem §3 |

---

## 1. Kiểm tra GPU

(Đã xử lý: trước đó cần reboot do driver tự cập nhật 580.173 → 580.178. Nếu sau này `nvidia-smi`
lại báo `Driver/library version mismatch` hoặc torch báo CUDA error 804 thì chỉ cần reboot.)

```bash
cd "/home/tung/Desktop/Lab Phuong Xuan Quang/AI/Research_Project_v1/modanet"
source ../.venv/bin/activate

nvidia-smi                           # phải hiện GTX 1050, không lỗi
python -m src.utils.device           # in cap=(6,1), arch list, chạy thử kernel conv thật
pytest tests/ -q                     # test CPU/GPU giờ sẽ chạy thay vì skip
```

`python -m src.utils.device` chạy **kernel thật** trên GPU. Bản wheel `torch 2.6.0+cu124` có thể
không ghi `sm_61` trong `get_arch_list()` mà chỉ có `sm_60`. Như vậy vẫn chạy được, vì cubin sm_60
tương thích nhị phân với sm_61. Chỉ khi dòng `kernel_test` báo `FAIL` mới cần đổi wheel:

```bash
pip uninstall -y torch
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. Train thủ công

### 2.1. Smoke test trên GPU (~2 phút)

```bash
python scripts/train.py --smoke --subset 20000 --epochs 2
```

Dòng cuối in `Thời gian/epoch đo được`. Lưu ý: tập con 20k mẫu nằm rải rác trên file, nên epoch 1
bị chậm do I/O (thực đo 618 ms/iter). Epoch 2 (cache ấm) phản ánh tốc độ GPU: 123 ms/iter.
Để so sánh, chạy trên CPU (i7-9700) mất ~27 phút/epoch.

**Thực đo trên GTX 1050 (2026-09-22):**

| Hạng mục | Giá trị |
|---|---|
| Tính toán fwd+bwd | 0.94 ms/mẫu, không đổi theo batch 32/64/128 |
| Batch | `--batch-size auto` (mặc định) → 32 × accum 4 = 128 trên GPU 2 GB (instruction 9.3). Batch 128 trực tiếp cũng vừa (~0.6 GB đỉnh), nếu muốn thì truyền `BS=128` |
| Đọc dữ liệu (HDD, `--shuffle chunk`) | ~970 mẫu/s, GPU cần ~1,065 mẫu/s → gần cân bằng |
| **1 epoch (324,086 mẫu train)** | **~5.5–7 phút** (smoke test: 21 s / 20k mẫu khi cache ấm; epoch đầu chậm hơn) |
| **60 epoch** | **~5.7–7 giờ** |

Muốn nhanh hơn thì chép `data_processed/` sang SSD (`/media/tung/FC0C…`, còn trống 218 GB), đặt
`MODANET_PROCESSED_DIR` hoặc dùng `--data-dir`, rồi thêm `--shuffle random`.

### 2.2. Train đầy đủ (cấu hình bài báo)

```bash
bash run_train.sh                  # nền + log; micro-batch theo VRAM, batch hiệu dụng 128
tail -f outputs/run01/train.out    # theo dõi
```

Lệnh tương đương khi chạy tay:

```bash
python scripts/train.py --preset paper --batch-size 32 --accum-steps 4 --out outputs/run01
```

| Tình huống | Lệnh |
|---|---|
| Mất điện / Ctrl+C / tắt máy | `bash run_train.sh` (tự resume), hoặc `python scripts/train.py --resume outputs/run01/last.pt` |
| Hết VRAM | `BS=16 bash run_train.sh` (accum tự thành 8, batch hiệu dụng vẫn 128) |
| Giới hạn thời gian mỗi phiên | `bash run_train.sh --max-hours 6` (dừng an toàn, chạy lại để tiếp) |
| Phép đo khắt khe (chia theo tổ hợp) | `OUT=outputs/run02 bash run_train.sh --split-mode group` |
| Dữ liệu trên SSD / đủ RAM | thêm `--shuffle random` (hoán vị ngẫu nhiên thuần) |

Mỗi epoch in một dòng như sau:

```
epoch       lr  train_loss  val_loss      gap  val_acc_amc  val_acc_doa   rmse°     time      ETA
   12   0.0100      1.8423    1.9011   +0.059        82.4%        61.2%    1.35    4m51s  3h47m
```

Mỗi epoch cũng ghi thêm một dòng vào `outputs/run01/log.csv` (flush + fsync), rồi lưu
`last.pt` + `best.pt` (theo val_loss). Checkpoint được ghi nguyên tử (tmp → rename) và chứa
epoch, model, optimizer, scheduler, trạng thái RNG, config snapshot (preset, hparams, args).

**Theo dõi overfitting (6B.3):** nhìn cột `gap = val_loss − train_loss`. Nếu `val_loss` tăng liên
tục 5 epoch trong khi `train_loss` giảm, script sẽ in cảnh báo. Early stopping (patience 8) bật
mặc định theo instruction 8.1. Muốn train đủ 60 epoch như MCNet thì thêm `--no-early-stop`
(xem §3, mục 3). Không tự thêm dropout / weight decay (bài gốc không có).

### 2.3. Inference

```bash
python scripts/inference.py --ckpt outputs/run01/best.pt --input ../MoDANet-dataset/SNR+10dB/QPSK/+25
python scripts/inference.py --ckpt outputs/run01/best.pt --input sample.npy        # (1024,5,2) hoặc (5,1024,2)
python scripts/inference.py --ckpt outputs/run01/best.pt --from-split test --num 32
python scripts/inference.py --ckpt outputs/run01/best.pt --benchmark 2000          # ms/frame (Fig.6b ~1.9 ms)
```

Softmax chỉ được áp ở đây. Model trả về logits.

---

## 3. MODANET_INSTRUCTION.md ↔ bài báo gốc: ƯU TIÊN instruction, chỉ ghi nhận

Đã đối chiếu từng con số của instruction với chữ, Bảng I và Fig. 2–6 của bài (xem ở 300 dpi), và
đối chiếu các siêu tham số [GAP] với bài MCNet. **Code theo instruction.** Các điểm dưới đây chỉ
ghi lại để tham khảo khi viết báo cáo.

**Khớp với bài:** Bảng I, thứ tự conv→pool→norm→relu, res-block 2+1 nhóm với phép add, Y-shape
Type A/B/C (Fig. 3a), output block conv→norm→relu→fc→softmax, loss L_AMC + L_DOA, K = 64, filter
size chốt 7×1, M = 5, N = 1024, DOA −60…60°, SNR −10…20 dB, chia 80/20, 450,120 mẫu, ~1.9 ms/frame
(Fig. 6b đo được ≈ 1.87 ms). Siêu tham số từ MCNet khớp nguyên văn: SGD momentum 0.9, lr 0.01
"dropped 90% after 30 epochs", batch 128, 60 epoch, "randomly initial weights".

**Khác với bài (không áp dụng):**

| # | Instruction (đang dùng) | Bài báo | Nếu muốn thử theo bài |
|---|---|---|---|
| 1 | Conv output block **luôn 1×1** (1.3) → 457,413 tham số | Đoạn chữ về Fig. 4b: *"The conv layers in the input block, the backbone flow of res-block, **and the output block** are tuned with different filter sizes"* → với k = 7 thì conv output là 7×1 | `MoDANet(output_conv_k=7)` → 506,565 tham số |
| 2 | Fig. 6b "~0.5–0.6×10⁶", nên 457,413 "khớp" (2.5) | Đo pixel Fig. 6b: thanh MoDANet ≈ **0.60×10⁶ (±0.01)**. 457k nằm dưới vạch 0.5 | Không có cấu hình k = 7 nào cho đúng 0.60M. Ứng viên: k = 9 và conv output 9×1 (605,509) |
| 3 | Early stopping patience 8 bật mặc định (8.1) | Bài không nói; MCNet train cố định 60 epoch, lr ×0.1 sau epoch 30. Nếu dừng trước epoch 30 thì mất pha lr = 0.001 | `--no-early-stop` (best.pt vẫn được chọn theo val_loss) |
| 4 | So sánh Type A/B/C (`scripts/evaluate.py`) chạy với `FILTER_SIZE = 7` (giá trị chốt cuối, giống mọi run khác trong repo) | Bảng I ghi filter size Table-I = 3×1; không rõ Fig. 3(b) — thí nghiệm Y-shape — chạy với k = 3 hay k = 7 (Fig. 4b tinh chỉnh k là thí nghiệm *sau* Fig. 3 trong thứ tự trình bày) | Muốn dựng đúng thứ tự bài (Fig. 3 trước, dùng k=3): train lại 3 checkpoint với `--filter-size 3` |

Kết quả đo được (`outputs/eval/per_snr.csv`, test set thật, k=7): AMC accuracy trung bình TypeA
93.89%, TypeB 92.61%, TypeC 91.92% — đúng thứ tự bài (A > B > C), hội tụ ≈99.5% tại SNR ≥ 10 dB.
Bài báo báo cáo Fig. 3(b) "accuracy > 85%, RMSE < 0.3° tại SNR ≥ 10 dB" cho thí nghiệm Y-shape,
thấp hơn nhiều so với 99%+ đo được ở đây — nhưng khớp với chính con số **99.41%** bài báo báo cáo ở
Fig. 6(a) cho cấu hình MoDANet cuối cùng (K=64, k=7, N=1024, TypeA) tại 10 dB. Giả thuyết: Fig. 3(b)
gốc chạy với k=3 (Bảng I), còn Fig. 6(a) mới dùng k=7 đã tinh chỉnh — repo này luôn dùng k=7 nên số
của TypeA khớp Fig. 6(a) hơn là Fig. 3(b). RMSE hội tụ về sàn ≈0.5°, khớp phát hiện MUSIC/DOA trước
đó (xem note phiên 2026-09-26 trong memory dự án), thấp hơn 0.017° bài báo báo cáo.

**Chi tiết MATLAB (không áp dụng):** input block có "an input layer"; `imageInputLayer` của MATLAB
mặc định trừ ảnh trung bình ('zerocenter'). Ảnh trung bình của tập train khác 0 thật (RMS 0.090),
đến từ QFSK/8FSK/LFM/DSB-SC/SSB-SC ở anten 0. Instruction chốt `NORMALIZE=False`, nên giữ nguyên.

## 4. MODANET_INSTRUCTION.md ↔ dữ liệu thật / môi trường

1. **Thứ tự tên điều chế theo chỉ số file ≠ thứ tự bài báo liệt kê.** Công thức giải mã tên
   file (PHẦN 6.3) đúng, nhưng `mod_i` ứng với
   `["16QAM","64QAM","PSK","QPSK","8PSK","QFSK","8FSK","16APSK","4PAM","LFM","DSB-SC","SSB-SC"]`,
   không phải `["PSK","QPSK",...]`. Bằng chứng:
   - Quét 450,120 file: mỗi thư mục `<MOD>` ứng với đúng 1 giá trị `mod_i` theo bảng trên
     (ví dụ `SNR-10dB/PSK/-60/` chứa `data00002421…30`, tức `mod_i = 2`).
   - Vật lý: cumulant C42 (bất biến pha) ở 20 dB. Nhóm PSK/QPSK/8PSK/QFSK/8FSK/SSB-SC ≈ −0.99
     (đường bao hằng), 16QAM −0.70 (lý thuyết −0.68), 64QAM −0.60 (lý thuyết −0.62). Vậy tên
     thư mục đúng.

   Mẫu 4PAM trong instruction khớp là do 4 lớp cuối trùng thứ tự. **Việc này không ảnh hưởng tới
   train** (chỉ là hoán vị tên lớp), nhưng sẽ ảnh hưởng confusion matrix / accuracy theo lớp nếu
   dùng sai bảng. Đã sửa trong `configs/data_config.py` (`MODULATIONS`) và có test khóa lại
   (`test_physics_modulation_names`). `PAPER_MODULATION_ORDER` được giữ chỉ để hiển thị.
2. **Dataset là cây thư mục** `SNR+xxdB/<MOD>/<±DOA>/dataXXXXXXXX.mat`, không phải thư mục phẳng.
   `convert_dataset.py` duyệt đệ quy và đối chiếu chéo nhãn tên file với tên thư mục
   (lệch 0/450,120).
3. **Công suất frame không ≈ 1 ở mọi SNR.** Instruction (đo 10 file 4PAM @3 dB) kết luận công suất
   0.99–1.23 và |x| ≤ 3.34. Trên dataset đầy đủ, công suất frame ≈ **1 + 10^(−SNR/10)**: tín hiệu
   có công suất đơn vị rồi mới cộng nhiễu. Median tỉ số là 1.008; ở −10 dB công suất ≈ 10.9; |x| max
   là 13.44 (4.81 nếu SNR ≥ 0). Vẫn giữ `NORMALIZE=False` (bài không chuẩn hóa, thang đo bị chặn,
   fp16 dư sức, BN ngay sau conv đầu). Hợp đồng dữ liệu đã được sửa theo SNR
   (`SIGNAL_POWER_RANGE`, `expected_frame_power`) và có test `test_processed_power_follows_snr`.
4. **GPU có 2 GB** → theo instruction 9.3 dùng batch 32 × accum 4. `sm_61` không có trong
   `get_arch_list()` của torch 2.6.0+cu124, nhưng sm_60 tương thích nhị phân và kernel test chạy OK,
   nên không cần đổi wheel.
5. **Thứ tự đọc dữ liệu (`--shuffle chunk`, mặc định).** `/home` nằm trên HDD, còn memmap
   9.2 GB lớn hơn RAM trống (~8 GB), nên đọc ngẫu nhiên thuần sẽ tốn ~40 phút I/O mỗi epoch.
   Sampler xáo theo khối 64 mẫu liền kề, gom 64 khối rồi xáo trộn bên trong. Mỗi batch 128 lấy
   từ ~55–64 vị trí ngẫu nhiên khác nhau (có test). Đây là chi tiết I/O `[GAP]`, không đổi model
   hay phân phối dữ liệu.
6. Chưa có `--weight-decay` / dropout: đúng như 6B.4, chỉ thêm khi có bằng chứng overfitting.

---

## 5. Cấu trúc

```
configs/model_config.py   ★ hằng số kiến trúc — nguồn chân lý, có nhãn [TABLE-I]/[TEXT]/[DERIVED]/[GAP]
configs/data_config.py    lớp, góc, SNR, đường dẫn, hợp đồng dữ liệu
configs/train_config.py   siêu tham số (toàn [GAP], theo MCNet)
src/models/blocks.py      ConvUnit, InputBlock, ResBlock, OutputBlock — không có số magic
src/models/modanet.py     MoDANet, MultiTaskLoss
src/data/labels.py        decode/encode tên file, nhãn thư mục
src/data/convert.py       scan + verify-index + .mat -> memmap (resume được)
src/data/dataset.py       ModaNetDataset (memmap, đọc theo batch) + ChunkShuffleBatchSampler
src/data/splits.py        chia random/group, lưu index một lần
src/data/physics.py       ước lượng DOA từ pha, C42 — kiểm vật lý
src/utils/                seed, device (kiểm GPU), checkpoint, metrics
scripts/                  convert_dataset, summary, train, inference, evaluate (stub), pack_for_server.sh
run_train.sh              train nền + tự resume | setup_server.sh: chuẩn bị máy mới
requirements.txt          chung mọi máy | requirements-gtx1050.txt: ghim torch 2.6.0 (Pascal)
tests/                    test từng block + khớp lớp + dữ liệu 3 tầng; fixtures = 22 file .mat thật
data_processed/           x_float16.npy, labels.npy, meta.json, splits/*.npz
```

**Quy tắc "không số magic":** `tests/test_config.py::test_no_magic_numbers_in_models` quét AST của
`src/models/*.py` và sẽ fail nếu có bất kỳ hằng số int/float nào. Mọi số phải import từ `configs/`.

### Đường dẫn

Mặc định dataset gốc nằm ở `../MoDANet-dataset`, memmap ở `./data_processed`. Có thể đổi bằng
biến môi trường `MODANET_DATA_ROOT`, `MODANET_PROCESSED_DIR`, hoặc tham số `--root/--out/--data-dir`.

### Convert lại (nếu cần)

```bash
python scripts/convert_dataset.py --verify-index     # chỉ kiểm (~4 s)
python scripts/convert_dataset.py --inspect          # thống kê + kiểm vật lý
python scripts/convert_dataset.py --dtype float16    # ~15–30 phút trên HDD, Ctrl+C rồi chạy lại để tiếp
```

---

## 6. Chạy trên máy khác / server vast.ai

### 6.1. Code tự thích nghi với máy

| Tham số | Mặc định `auto` làm gì |
|---|---|
| `--device auto` | CUDA (chạy thử kernel thật) → Apple MPS → CPU. Chọn GPU cụ thể: `--device cuda:1` hoặc `CUDA_VISIBLE_DEVICES=1` |
| `--batch-size auto` | micro-batch theo VRAM: < 1.5 GB → 16; 2 GB → 32 (9.3); 4 GB → 64 (9.3); ≥ 6 GB → 128. `accum = 128 / micro`, nên **batch hiệu dụng luôn là 128** |
| `--num-workers auto` | `min(4, số CPU − 1)`, tự hạ nếu `/dev/shm` nhỏ. Docker mặc định chỉ có 64 MB, dễ gây lỗi "Bus error" |
| `--in-memory` | nạp 9.2 GB vào RAM, bỏ hẳn I/O đĩa. Nhanh nhất trên server; tự từ chối nếu RAM không đủ. Linux: vẫn dùng worker như trên (fork, các worker đọc chung một bản dữ liệu trong RAM). Windows/macOS/Python ≥ 3.14: 0 worker, vì mỗi worker sẽ phải chép 9.2 GB |
| `--resume` | resume được **trên máy khác**: micro-batch tính lại theo GPU mới, batch hiệu dụng và lịch lr giữ nguyên. Ví dụ đang train trên GTX 1050 (32×4) có thể chuyển sang server (128×1) |

Các lỗi đã sửa để chạy được ở mọi nơi:
- **Windows/macOS/Python ≥ 3.14:** DataLoader dùng `spawn`/`forkserver`, trước đây mỗi worker bị chép nguyên memmap 9.2 GB. Giờ chỉ gửi đường dẫn (pickle ~2 MB).
- **GPU Ampere trở lên (RTX 30/40/50, A100, H100):** cuDNN mặc định dùng TF32, nên test so CPU/GPU tắt TF32 trước khi so ngưỡng 1e-4.
- Resume được trên máy có số GPU khác.
- Không còn đường dẫn tuyệt đối nào trong code (có test kiểm tra).
- **GPU nhanh (RTX 4090…):** trước đây `--in-memory` chạy 0 worker, và vòng train gọi `.item()` mỗi iteration.
  CPU phải chuẩn bị xong batch (~6 ms/batch 128 trên i7-9700) rồi GPU mới chạy tiếp. Giờ worker chuẩn bị
  batch song song, loss được cộng dồn trên GPU và chỉ đọc về khi in log. `log.csv` và trọng số giống hệt
  trước khi sửa (đã so bằng smoke `--deterministic` trên CPU).

### 6.2. Chọn máy trên vast.ai

- **Template:** PyTorch (CUDA ≥ 12.x). torch có sẵn trong image, nên không cần cài lại.
- **Disk ≥ 30 GB** (dữ liệu 9.2 GB + image + checkpoint). **RAM ≥ 16 GB** để dùng `--in-memory`.
- **GPU:** model nhỏ (457k tham số), nên RTX 3060/3090/4090 là đủ; GPU lớn hơn không giúp nhiều. Ước lượng chưa đo: nhanh hơn GTX 1050 khoảng 10–20 lần, tức 60 epoch trong khoảng 0.5–1 giờ với `--in-memory`.
- RTX 50xx (Blackwell) cần torch ≥ 2.7 bản cu128. `python -m src.utils.device` sẽ báo nếu bản torch không hỗ trợ GPU.

### 6.3. Đưa code + dữ liệu lên (thay `<PORT>`, `<IP>` bằng thông tin SSH của instance)

Trên máy này:
```bash
cd "/home/tung/Desktop/Lab Phuong Xuan Quang/AI/Research_Project_v1/modanet"
bash scripts/pack_for_server.sh                       # -> dist/modanet_code.tar.gz (~0.9 MB)
scp -P <PORT> dist/modanet_code.tar.gz root@<IP>:/workspace/
ssh -p <PORT> root@<IP> "cd /workspace && tar -xzf modanet_code.tar.gz"
# dữ liệu đã convert (9.2 GB); rsync chạy tiếp được nếu mạng đứt (--partial)
rsync -avP --partial -e "ssh -p <PORT>" data_processed/ root@<IP>:/workspace/modanet/data_processed/
```
`data_processed/` gồm `x_float16.npy`, `labels.npy`, `meta.json`, `splits/random_seed42.npz` (giữ đúng
cách chia tập như máy này) và `SHA256SUMS`.

*Sửa code rồi đồng bộ lại:* `bash scripts/sync_code.sh`, hoặc **Ctrl+Shift+B** trong VS Code
(`.vscode/tasks.json` ở thư mục gốc workspace). Lệnh chỉ gửi file đã đổi; code trên server giống hệt
local (file thừa bị xoá), còn `data_processed/`, `outputs/`, `*.pt` trên server được giữ nguyên.
Cần khối `Host vast` trong `~/.ssh/config`, hoặc chỉ định trực tiếp:
`HOST=root@<IP> PORT=<PORT> bash scripts/sync_code.sh`. Thêm `-n` để chạy thử trước.

*Cách khác, nếu mạng upload chậm:* đưa bộ `.mat` gốc (hoặc file zip 17 GB) lên, rồi chạy
`python scripts/convert_dataset.py --root /workspace/MoDANet-dataset --workers 16` trên server.
Cách này sẽ tạo lại splits với cùng seed 42.

### 6.4. Trên server

```bash
ssh -p <PORT> root@<IP>
cd /workspace/modanet
bash setup_server.sh                                   # cài thư viện, kiểm GPU, kiểm checksum, chạy test
python scripts/train.py --smoke --subset 20000 --epochs 2   # đo thời gian/epoch
bash run_train.sh --in-memory                          # train đầy đủ ở nền (mất SSH vẫn chạy)
tail -f outputs/run01/train.out
```

Instance bị dừng hoặc gián đoạn thì chạy lại `bash run_train.sh --in-memory`: script tự resume từ
`last.pt` (checkpoint được lưu mỗi epoch).

### 6.5. Lấy kết quả về máy này

```bash
rsync -avP -e "ssh -p <PORT>" root@<IP>:/workspace/modanet/outputs/ ./outputs/
python scripts/inference.py --ckpt outputs/run01/best.pt --from-split test --num 32   # chạy được cả trên CPU
```
Checkpoint train trên GPU nạp được trên CPU, GTX 1050 hay GPU bất kỳ (`map_location='cpu'`).

### 6.6. Server chạy Arch Linux

Code không phụ thuộc bản phân phối Linux: thuần Python + PyTorch/NumPy/SciPy, chỉ dùng `/proc/meminfo`,
`/dev/shm`, `bash`, `sha256sum`, `nohup` (Linux nào cũng có). Khác biệt nằm ở **bước cài đặt**.
*(Chưa chạy thử trên máy Arch thật, các ghi chú dưới đây rút ra từ việc đọc code và script.)*

```bash
sudo pacman -S --needed python rsync                   # + driver, xem mục "Driver" bên dưới
cd ~/modanet
python -m venv ../.venv                                # venv NGOÀI modanet/ (giống máy này), xem lưu ý 1
PY=../.venv/bin/python bash setup_server.sh            # cài thư viện, kiểm GPU, checksum, chạy test
bash run_train.sh --in-memory                          # run_train.sh tự tìm ../.venv/bin/python
```

Lưu ý:

1. **Không `pip install` vào Python hệ thống được.** Arch chặn (lỗi `externally-managed-environment`),
   nên `bash setup_server.sh` chạy thẳng sẽ dừng ở bước 1. Phải dùng venv. Đặt venv ở `../.venv`,
   **không** đặt trong `modanet/`: `scripts/sync_code.sh` chạy `rsync --delete` nên sẽ xoá mọi file trên
   server không có ở local, kể cả `modanet/.venv`.
2. **Python của Arch rất mới (3.13 / 3.14, kiểm bằng `python --version`).**
   - `requirements.txt`: không sao, pip lấy bản torch mới có wheel cho Python đó.
   - `requirements-gtx1050.txt` (chỉ cần nếu server là GPU Pascal GTX 10xx): `torch==2.6.0` và
     `numpy<2` không có wheel cho Python mới, nên cần venv Python 3.12:
     ```bash
     sudo pacman -S uv
     uv venv --seed --python 3.12 ../.venv                # --seed: có pip, vì setup_server.sh gọi `$PY -m pip`
     ```
3. **Python 3.14 + `--in-memory` chạy 0 worker.** Từ 3.14, Linux mặc định tạo worker bằng `forkserver`
   thay vì `fork`, nên code tự đặt `num_workers = 0` để không chép 9.2 GB sang từng worker
   (`src/utils/device.py`, `resolve_num_workers`). Kết quả vẫn đúng, nhưng CPU có thể thành nút cổ chai
   trên GPU nhanh. Tránh bằng venv Python ≤ 3.13 (lệnh `uv` ở trên, đổi `3.12` thành `3.13` nếu không cần GTX 10xx).
4. **Driver NVIDIA.** torch từ pip đã kèm CUDA runtime, nên chỉ cần driver (`nvidia-utils` + gói kernel
   module), không cần gói `cuda` của Arch.
   - Arch cập nhật liên tục: sau mỗi `pacman -Syu` có cập nhật driver phải **khởi động lại**, không thì
     gặp lỗi CUDA 804 / `nvidia-smi: Driver/library version mismatch` (máy này từng bị).
   - GPU Pascal (GTX 10xx) hoặc cũ hơn: driver mới đã bỏ hỗ trợ, phải dùng nhánh 580 từ AUR
     (`nvidia-580xx-dkms`). Kiểm lại trên Arch Wiki trước khi cài.
5. **Đường dẫn khác vast.ai.** Các lệnh ở §6.3–6.5 dùng `root@<IP>` và `/workspace/`. Trên Arch, đổi thành
   user của bạn và thư mục nhà, ví dụ:
   ```bash
   rsync -avP --partial data_processed/ user@<IP>:modanet/data_processed/
   HOST=user@<IP> DEST=modanet bash scripts/sync_code.sh   # DEST tương đối = ~/modanet trên server
   ```
   Không viết `DEST=~/modanet`: shell ở máy local sẽ đổi `~` thành `/home/tung/...`. `rsync` phải được cài
   ở **cả hai** máy.

### 6.7. Máy Windows / macOS
Không dùng được các script `.sh`; gọi Python trực tiếp: `python scripts/train.py ...`,
`python -m src.utils.device`, `python -m pytest -q`. Đặt đường dẫn dữ liệu bằng `--data-dir`
hoặc biến môi trường `MODANET_PROCESSED_DIR`.

### 6.8. Server Debian (vd. laptop GTX 1060 6 GB dùng làm máy remote)

GTX 1060 cùng kiến trúc Pascal (sm_61) với GTX 1050 ở máy này (tên file `requirements-gtx1050.txt`
chỉ là lịch sử — điều kiện thật là "GPU Pascal sm_6x", không phải đúng model 1050), nên vẫn cần
ghim `torch==2.6.0` (Pascal bị bỏ ở bản torch mới, đã gặp lỗi CUDA "driver too old" khi cài
thiếu bước ghim này trên máy thật 2026-09-29). **Đã chạy thành công trên GTX 1060 Max-Q 6 GB
thật (2026-09-29):** VRAM báo về chỉ **5.92 GB** (Max-Q bớt so với 6 GB danh nghĩa), dưới ngưỡng
`≥ 6 GB` của bảng `--batch-size auto` (§6.1), nên tự chọn **64 × accum 2 = 128** (batch hiệu
dụng vẫn 128) chứ không phải 128×1 như GPU 6 GB đúng nghĩa — không cần bạn chỉnh tay, chỉ cần
biết đây không phải dấu hiệu lỗi.

```bash
# Trên máy Debian (có sudo)
sudo apt update
sudo apt install -y build-essential python3-venv python3-pip rsync git
```

**Driver NVIDIA** — Debian không kèm sẵn driver độc quyền, phải bật kho `contrib non-free
non-free-firmware` trước (gói `nvidia-driver` nằm ở `non-free`, không phải chỉ
`non-free-firmware` — thiếu `non-free` sẽ ra lỗi "has no installation candidate", đã gặp trên
máy thật 2026-09-29). **Kiểm định dạng file trước khi sửa** — cài từ ISO netinst trixie thường
vẫn là `sources.list` dạng cũ (`deb http://deb.debian.org/debian/ trixie main
non-free-firmware`, chưa hẳn deb822 `/etc/apt/sources.list.d/debian.sources` như bản mặc định
mới); `cat` cả hai đường dẫn để biết máy bạn dùng cái nào:

```bash
cat /etc/apt/sources.list.d/debian.sources 2>/dev/null
cat /etc/apt/sources.list 2>/dev/null
```

Dạng cũ (`sources.list` có dòng `deb ... trixie main non-free-firmware`, xác nhận trên máy thật
2026-09-29):

```bash
sudo sed -i 's/ main non-free-firmware/ main contrib non-free non-free-firmware/' \
  /etc/apt/sources.list
cat /etc/apt/sources.list          # kiểm: các dòng deb/deb-src chính phải có đủ 4 từ sau "trixie"
```

Dạng deb822 (`debian.sources`, có dòng `Components: main non-free-firmware` hoặc chỉ `main`):

```bash
sudo sed -i 's/^Components: main non-free-firmware$/Components: main contrib non-free non-free-firmware/;
              s/^Components: main$/Components: main contrib non-free non-free-firmware/' \
  /etc/apt/sources.list.d/debian.sources
cat /etc/apt/sources.list.d/debian.sources
```

Rồi cài driver:

```bash
sudo apt update
sudo apt install -y nvidia-driver firmware-misc-nonfree
sudo reboot                          # BẮT BUỘC sau khi cài/đổi driver (giống lưu ý 4 ở §6.6)
nvidia-smi                           # phải hiện GTX 1060, không lỗi
```

Nếu `nvidia-smi` báo "couldn't communicate with the NVIDIA driver" dù gói đã cài (đã gặp trên
máy thật 2026-09-29): kiểm `sudo dkms status` — nếu ra `nvidia-current/<version>: added` (chưa
phải `installed`), nghĩa là module DKMS mới đăng ký chứ chưa build cho kernel đang chạy, thường
do thiếu `linux-headers` đúng bản lúc cài gói. Sửa bằng:

```bash
sudo apt install -y linux-headers-$(uname -r)
sudo dkms build nvidia-current/<version> -k $(uname -r)     # <version> lấy từ dkms status
sudo dkms install nvidia-current/<version> -k $(uname -r)
sudo modprobe nvidia && nvidia-smi
```

(Không liên quan Secure Boot nếu `mokutil --sb-state` báo `SecureBoot disabled`.)

**Lưu ý riêng cho laptop (Optimus/hybrid GPU):** nếu `nvidia-smi` không thấy GPU dù driver đã
cài xong, nhiều laptop tắt hẳn card rời theo mặc định (chuyển mạch NVIDIA Optimus). Kiểm tra:
- BIOS/UEFI có tuỳ chọn "Discrete/NVIDIA Graphics" hay "Hybrid" — chuyển sang **Discrete** nếu
  có, đơn giản nhất cho máy chạy headless qua SSH.
- Nếu chỉ có chế độ hybrid: cài `nvidia-prime`, chạy `sudo prime-select nvidia` rồi reboot lại.
  CUDA compute (không cần X) thường không cần PRIME, chỉ cần module kernel nvidia đã nạp
  (`lsmod | grep nvidia`) và `/dev/nvidia0` tồn tại.

**Python & venv** — Debian 12 "bookworm" có sẵn Python 3.11, tương thích thẳng
`torch==2.6.0`/`numpy<2` (`python3 -m venv .venv` là đủ). **Debian 13 "trixie" mặc định Python
3.13** (xác nhận trên máy thật 2026-09-29, `uname -a` → `deb13`): torch 2.6.0 có wheel cho 3.13,
nhưng `numpy<2` (ghim theo instruction 9.1b) **không có wheel cho 3.13** — pip sẽ cố build từ
nguồn (cần gfortran/BLAS, dễ lỗi hoặc rất chậm). Cần venv Python 3.12 riêng, giống Arch §6.6 lưu
ý 2, dùng `uv` (không cần gói Python 3.12 từ apt — trixie đã bỏ):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"        # hoặc mở lại shell — installer in đúng dòng cần chạy

cd ~
uv venv --seed --python 3.12 .venv   # --seed: có pip sẵn, vì setup_server.sh gọi `$PY -m pip`
                                      # venv NGOÀI modanet/, lý do giống §6.6 lưu ý 1
                                      # (sync_code.sh --delete sẽ xoá modanet/.venv nếu đặt trong đó)
mkdir -p modanet
```

(Nếu `python3 --version` trên máy bạn ra 3.11/3.12 thay vì 3.13 thì bỏ qua bước `uv`, dùng thẳng
`python3 -m venv .venv`.)

Đưa code + dữ liệu lên theo §6.3 (thay `root@<IP>` bằng user Debian của bạn, xem §6.6 lưu ý 5),
rồi cài đặt + kiểm tra:

```bash
cd ~/modanet
../.venv/bin/pip install -r requirements-gtx1050.txt   # ghim torch==2.6.0 + numpy<2 cho Pascal — BẮT
                                                         # BUỘC chạy dòng này TRƯỚC setup_server.sh, nếu
                                                         # không `pip install -r requirements.txt` (chung)
                                                         # trong script sẽ tự lấy torch mới nhất, không
                                                         # tương thích Pascal (đã gặp trên máy thật)
PY=../.venv/bin/python bash setup_server.sh             # kiểm GPU, checksum dữ liệu, chạy test
../.venv/bin/python scripts/train.py --smoke --subset 20000 --epochs 2   # đo thời gian/epoch thật
bash run_train.sh --in-memory                            # train đầy đủ ở nền, tự resume
tail -f outputs/run01/train.out
```

`run_train.sh` tự tìm `../.venv/bin/python` (giống §6.6), nên không cần đặt lại `PY=` ở bước
này. Bỏ `--in-memory` nếu RAM máy < 12 GB. Lấy kết quả về theo §6.5.

### 6.9. Server GPU A100 / H100 (VRAM lớn)

Phần này chỉ nói đặc thù **phần cứng** A100 (Ampere, sm_80) / H100 (Hopper, sm_90); phần cài hệ
điều hành / driver / venv thì theo OS thật của server: Ubuntu/Debian → §6.8, Arch → §6.6,
vast.ai template → §6.2–§6.4.

**Không cần ghim torch.** Khác GTX 10xx (Pascal), A100/H100 được mọi bản torch gần đây hỗ trợ
sẵn, nên dùng thẳng `requirements.txt` — **không** dùng `requirements-gtx1050.txt`:

```bash
pip install -r requirements.txt      # bỏ qua nếu image cloud đã có sẵn torch (vd. template PyTorch vast.ai)
python -m src.utils.device           # kiểm: phải hiện tên A100/H100, kernel_test OK, VRAM ~40/80 GB
```

**VRAM 40/80 GB không làm batch lớn hơn.** Model chỉ 457,413 tham số, nên `--batch-size auto`
(bảng §6.1) đã chọn mức cao nhất ngay từ ngưỡng `≥ 6 GB`: micro-batch **128 × accum 1 = 128**.
A100 40 GB hay 80 GB đều rơi vào đúng ngưỡng đó — không có ngưỡng cao hơn trong
`AUTO_BATCH_BY_VRAM_GB` (`configs/train_config.py`), vì batch hiệu dụng bị khoá cứng ở 128 theo
MODANET_INSTRUCTION.md 9.3. Không cần tự chỉnh `--batch-size`/`--accum-steps`.

**Nút cổ chai thực tế là CPU/I-O, không phải GPU compute** — GPU này nhanh hơn GTX 1050 rất
nhiều lần trong khi model lại rất nhỏ, nên DataLoader dễ thành nơi chờ. Hai điều nên làm:

- Bật `--in-memory` (RAM instance A100 cloud thường ≥ 40 GB, dư sức chứa 9.2 GB dữ liệu), và đổi
  `--shuffle random` vì dữ liệu đã ở RAM — không còn lý do dùng `chunk` (chế độ đó chỉ tối ưu cho
  đọc tuần tự trên HDD, xem comment `SHUFFLE_MODE` trong `configs/train_config.py`).
- `--num-workers auto` bị khoá trần ở `min(4, số CPU − 1)` (`NUM_WORKERS = 4` trong
  `configs/train_config.py`, đặt theo máy 8 nhân của tác giả) — **không** tự tăng theo số vCPU
  của server cloud. Nếu `python -m src.utils.device` báo ≥ 16 CPU, truyền tay
  `--num-workers 8` (hoặc cao hơn, thử tăng dần) thay vì để `auto`.

```bash
python scripts/train.py --device auto --batch-size auto --in-memory --shuffle random \
  --num-workers 8 --preset paper --out outputs/run01
# hoặc nền + tự resume (run_train.sh forward mọi tham số thêm sau nó):
bash run_train.sh --in-memory --shuffle random --num-workers 8
```

**TF32:** cuDNN trên Ampere+ mặc định bật TF32 cho matmul/conv. Test so CPU/GPU đã tắt TF32 khi
so ngưỡng (§6.1) nên không ảnh hưởng kết quả kiểm thử; lúc train thì không cần chỉnh gì thêm.

**Multi-GPU:** code hiện chỉ dùng 1 GPU. `--device cuda:N` chọn đúng 1 card qua
`CUDA_VISIBLE_DEVICES`; nếu server có nhiều A100 thì chỉ tận dụng được 1 card — code không có
DDP/DataParallel.
