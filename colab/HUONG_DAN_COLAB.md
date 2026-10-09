# Chạy MoDANet trên Google Colab

Thư mục này có:

| File | Nội dung |
|---|---|
| `MoDANet_colab.ipynb` | Toàn bộ code (configs, model, dữ liệu, train, đánh giá, suy luận) trong **một notebook**, không cần upload thư mục code |
| `HUONG_DAN_COLAB.md` | Hướng dẫn này |

Model và siêu tham số giống hệt bản local, theo `MODANET_INSTRUCTION.md`: Type A, K = 64, filter 7×1,
conv output 1×1, **457,413 tham số**, SGD momentum 0.9, lr 0.01 (×0.1 sau epoch 30), batch hiệu dụng
128, 60 epoch, early stopping patience 8. Khi chạy, notebook tự kiểm lại số tham số của mọi cấu hình
so với giá trị vàng. Checkpoint **dùng chung hai chiều** với máy local (xem mục 6).

---

## 1. Đưa dữ liệu lên Google Drive (làm một lần)

Colab không đọc được ổ cứng của máy bạn, nên phải đưa dữ liệu **đã convert** lên Google Drive:

```
MyDrive/modanet/data_processed/
├── x_float16.npy            9.22 GB
├── labels.npy               2.7 MB
├── meta.json
├── SHA256SUMS
└── splits/random_seed42.npz
```

Tổng ~9.3 GB, vừa gói Drive miễn phí 15 GB. Nên upload cả `splits/random_seed42.npz` để chia tập
train/val/test giống hệt máy local. Không upload thì notebook tự tạo lại với seed 42, cũng ra cùng cách chia.

Chọn **một** trong các cách sau.

### Cách A — rclone, chia nhỏ file (khuyên dùng khi mạng hay đứt)

rclone upload lại được: file nào đã lên đủ thì bỏ qua. Nhưng một file 9.2 GB bị đứt giữa chừng thì
phải upload lại từ đầu, nên hãy cắt nó thành các phần 1 GB. Notebook tự ghép các phần lại.

```bash
# 1) Cài rclone và kết nối Google Drive (một lần)
sudo apt install rclone          # hoặc: curl https://rclone.org/install.sh | sudo bash
rclone config                    # n (new) -> name: gdrive -> storage: drive -> Enter qua các câu hỏi
                                 # -> scope 1 (full access) -> "Use web browser": y -> đăng nhập Google

# 2) Cắt file lớn thành các phần 1 GB (~1 phút; tạo thêm 9.2 GB tạm trên đĩa)
cd "/home/tung/Desktop/Lab Phuong Xuan Quang/AI/Research_Project_v1/modanet"
mkdir -p /tmp/modanet_upload/splits
split -b 1G data_processed/x_float16.npy /tmp/modanet_upload/x_float16.npy.part_
cp data_processed/{labels.npy,meta.json,SHA256SUMS} /tmp/modanet_upload/
cp data_processed/splits/random_seed42.npz /tmp/modanet_upload/splits/

# 3) Upload. Mạng đứt thì chạy lại đúng lệnh này: các phần đã xong được bỏ qua
rclone copy -P --transfers 2 /tmp/modanet_upload gdrive:modanet/data_processed

# 4) Kiểm tra trên Drive đủ 10 phần (part_aa … part_aj) + 4 file nhỏ, rồi xoá bản tạm
rclone ls gdrive:modanet/data_processed
rm -rf /tmp/modanet_upload
```

> Nếu `/tmp` không đủ 9.2 GB trống (`df -h /tmp`), đổi `/tmp/modanet_upload` thành một thư mục trên
> ổ SSD `/media/tung/FC0C…`.

### Cách B — rclone, upload nguyên file (mạng ổn định)

```bash
cd "/home/tung/Desktop/Lab Phuong Xuan Quang/AI/Research_Project_v1/modanet"
rclone copy -P data_processed gdrive:modanet/data_processed \
    --include "x_float16.npy" --include "labels.npy" --include "meta.json" \
    --include "SHA256SUMS" --include "splits/random_seed42.npz"
```

### Cách C — trình duyệt

Mở drive.google.com → tạo thư mục `modanet` → vào thư mục → **New → Folder upload** → chọn
`data_processed`. Đơn giản nhất, nhưng upload 9.2 GB qua trình duyệt dễ hỏng nếu mạng chập chờn.
Có thể xoá `progress.json` (nếu có) trên Drive sau khi upload vì notebook không dùng.

**Dù dùng cách nào**, notebook cũng kiểm SHA256 sau khi chép, nên file bị hỏng khi upload sẽ bị phát
hiện ngay (báo `SAI x_float16.npy`).

### (Hiếm khi cần) Không có `data_processed/`, chỉ có file zip `.mat` gốc

Đặt zip (~17 GB) vào `MyDrive/modanet/MoDANet-dataset.zip` và sửa trong notebook
`DATA_SOURCE = "mat_zip"`. Notebook sẽ giải nén rồi convert ngay trên Colab (mục 5B, ~1 giờ), rồi lưu
`data_processed/` lên Drive cho các lần sau. Cách này cần Drive ≥ 30 GB (gói miễn phí không đủ).

---

## 2. Mở notebook trên Colab

1. Vào https://colab.research.google.com → **File → Upload notebook** → chọn `modanet/colab/MoDANet_colab.ipynb`.
   (Hoặc upload file `.ipynb` vào Drive, nhấp đúp rồi chọn **Open with → Google Colaboratory**.)
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
   Bỏ bước này thì notebook sẽ chạy trên CPU và rất chậm.
3. Xem ô **1. CẤU HÌNH**. Nếu dữ liệu nằm đúng `MyDrive/modanet/data_processed/` thì không cần sửa gì.

---

## 3. Chạy

**Runtime → Run all** (Ctrl+F9). Khi Colab hỏi quyền truy cập Google Drive, chọn tài khoản rồi **Allow**.

Notebook sẽ lần lượt:

| Mục | Làm gì | Thời gian |
|---|---|---|
| 0 | In tên GPU (T4, 15 GB) | vài giây |
| 2 | Mount Drive, chép dữ liệu sang `/content`, ghép các phần, kiểm SHA256 | ~5–15 phút (chỉ lần đầu mỗi máy Colab) |
| 4 | Dựng model, kiểm số tham số = 457,413 | vài giây |
| 8 | **Smoke test**: 1 epoch trên 20,000 mẫu, in `Thời gian/epoch đo được` | vài phút |
| 9 | **Train đầy đủ** (60 epoch, hoặc dừng sớm theo early stopping) | xem bên dưới |
| 10 | Vẽ đường cong loss / accuracy / RMSE theo epoch | vài giây |
| 11 | Đánh giá `best.pt` trên tập test: accuracy AMC, accuracy DOA, RMSE theo SNR, confusion matrix | ~1–2 phút |
| 12 | In dự đoán cho 16 mẫu test, đo ms/frame | vài giây |

**Ước lượng thời gian train (chưa đo trên T4):** T4 tính nhanh hơn GTX 1050 khoảng 4–5 lần, tức
khoảng 1.5–3 phút mỗi epoch, 60 epoch mất khoảng 2–3 giờ. Colab free chỉ có 2 CPU nên khâu đọc dữ liệu
có thể làm chậm hơn. Con số thật là dòng `Thời gian/epoch đo được` của smoke test nhân với ~16
(324,086 / 20,000 mẫu).

Mỗi epoch in một dòng:
```
epoch       lr  train_loss  val_loss      gap  val_acc_amc  val_acc_doa   rmse°     time      ETA
   12   0.0100      1.8423    1.9011   +0.059        82.4%        61.2%    1.35    1m51s  1h29m  *best
```

Mọi kết quả được ghi thẳng lên Drive, trong `MyDrive/modanet/outputs/run01/`:
`log.csv`, `last.pt`, `best.pt`, `train_curves.png`, `test_results.json`, `test_per_snr.csv`,
`test_vs_snr.png`, `test_confusion_amc.png`.

---

## 4. Bị ngắt kết nối / hết giờ → chạy tiếp

Giới hạn của Colab free: phiên bị ngắt sau ~90 phút không tương tác, dài tối đa ~12 giờ, và đôi khi
báo hết quota GPU (*"cannot connect to GPU backend"*, phải chờ vài giờ).

- Mỗi epoch được lưu `last.pt` lên Drive, nên bị ngắt chỉ mất epoch đang chạy dở.
- Để chạy tiếp: mở lại notebook → chọn T4 → **Runtime → Run all**. Notebook chép lại dữ liệu (máy mới)
  và **tự resume** từ `outputs/run01/last.pt`. Có thể đặt `RUN_SMOKE = False` để bỏ bước smoke.
- `MAX_HOURS = 11.0`: train tự dừng an toàn trước khi phiên 12 giờ hết hạn.
- Muốn dừng tay: bấm ■ (Interrupt execution). Epoch đang chạy bị bỏ, các epoch đã xong vẫn còn.
- Run đã xong (đủ epoch hoặc early stopping) thì chạy lại sẽ **không** train thêm. Muốn train lần mới,
  đổi `RUN_NAME = "run02"`.
- Giữ tab trình duyệt mở trong lúc train. Colab Pro có chế độ chạy nền.

---

## 5. Các tuỳ chọn trong ô CẤU HÌNH

| Biến | Mặc định | Khi nào đổi |
|---|---|---|
| `DRIVE_DIR` | `/content/drive/MyDrive/modanet` | dữ liệu nằm ở thư mục khác trên Drive |
| `RUN_NAME` | `run01` | train thêm một lần chạy khác (`run02`, …) |
| `RUN_SMOKE` / `RUN_FULL` / `RUN_EVAL` | `True` | tắt bước không cần |
| `SPLIT_MODE` | `random` | `group`: phép đo khắt khe hơn (tổ hợp mod/DOA/SNR của test chưa gặp khi train). Cần đổi `RUN_NAME` |
| `NO_EARLY_STOP` | `False` | `True`: train đủ 60 epoch như MCNet (README §3, mục 3) |
| `IN_MEMORY` | `auto` | Colab Pro High-RAM tự nạp 9.2 GB vào RAM. Colab free (12.7 GB) đọc từ đĩa `/content` để tránh crash vì hết RAM |
| `SHUFFLE` | `chunk` | `random` nếu `IN_MEMORY` bật (hoán vị thuần) |
| `BATCH_SIZE` | `auto` | T4/L4/A100 → micro-batch 128 × accum 1. Batch hiệu dụng luôn là 128 |
| `MAX_HOURS` | `11.0` | Colab Pro (24 giờ) → tăng lên |

Không đổi được qua ô cấu hình (cố ý, theo instruction): kiến trúc, lr, momentum, lịch lr, loss.

---

## 6. Dùng chung checkpoint với máy local

`MoDANet_colab.ipynb` dùng đúng tên tham số, định dạng checkpoint và file chia tập của `scripts/train.py`.

**Colab → máy local:** tải `MyDrive/modanet/outputs/run01/` về `modanet/outputs/run01/` (hoặc dùng cell
cuối của notebook để tải file zip, không gồm `last.pt`), rồi chạy:
```bash
python scripts/inference.py --ckpt outputs/run01/best.pt --from-split test --num 32
python scripts/train.py --resume outputs/run01/last.pt      # train tiếp ở máy local nếu chưa xong
```

**Máy local → Colab:** đã train được vài epoch trên GTX 1050? Upload `outputs/run01/last.pt` và
`log.csv` vào `MyDrive/modanet/outputs/run01/` rồi chạy notebook. Notebook resume từ epoch đó, micro-batch
tính lại theo T4 (128 × 1 thay cho 32 × 4) nhưng batch hiệu dụng 128 và lịch lr giữ nguyên.

---

## 7. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `NVIDIA-SMI has failed` / `GPU: KHÔNG CÓ` | Runtime → Change runtime type → T4 GPU, rồi Run all lại |
| `Cannot connect to GPU backend` | Hết quota GPU miễn phí trong ngày. Chờ vài giờ, hoặc dùng Colab Pro |
| `Không thấy .../x_float16.npy` | Sai `DRIVE_DIR`, hoặc dữ liệu chưa upload xong. Kiểm tra trên Drive: `MyDrive/modanet/data_processed/` |
| `SAI x_float16.npy` khi kiểm SHA256 | File hỏng khi upload. Xoá trên Drive rồi upload lại (cách A: chỉ phần bị hỏng, chạy lại `rclone copy`), sau đó `!rm -rf /content/data_processed` và Run all |
| `Your session crashed after using all available RAM` | Đặt `IN_MEMORY = False` |
| `No space left on device` | Đĩa `/content` đầy (thường do `DATA_SOURCE = "mat_zip"`). Xoá `/content/MoDANet-dataset` sau khi convert |
| `CUDA out of memory` | Đặt `BATCH_SIZE = 64` (accum tự thành 2, batch hiệu dụng vẫn 128) |
| Chạy lại nhưng không train thêm | Run đã xong. Đổi `RUN_NAME` nếu muốn train lần mới |
