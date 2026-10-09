 # Phân tích dataset MoDANet — cấu trúc, cách sinh, khả năng phân tách DOA

*Cập nhật: 2026-09-26. Phạm vi: dataset gốc 450,120 mẫu dùng trong Doan et al., "MoDANet",
IEEE Commun. Lett. 26(2), 2022. Tài liệu này chỉ ghi nhận — không thay đổi code trong `modanet/`
(MODANET_INSTRUCTION.md vẫn là nguồn chân lý cho phần dựng lại mô hình).*

---

## 0. Tóm tắt

1. **Dataset được sinh khác với mô tả trong bài báo.** Metadata của chính tác giả
   (`myData_info.mat` trong repo [doans/MoDANet](https://github.com/doans/MoDANet)) cho thấy kênh
   đa đường chỉ gồm **2 profile cố định** (PedA và VehA bỏ 1 đường), không phải "1–10 đường NLOS
   với suy hao [−50, −1] dB và trễ [1, 3000] ns ngẫu nhiên". Chỉ góc NLOS là ngẫu nhiên (đều trong
   [−60°, 60°]). Metadata khớp dữ liệu tới sai khác ~10⁻⁵.
2. **DOA tách được độc lập với điều chế:** độ chính xác DOA của MUSIC gần như không đổi giữa 12
   điều chế (64–70% đúng lớp ở SNR ≥ 10 dB).
3. **Có sai số sàn ≈ 0.52° (≈ 67% đúng lớp 1°) với phương pháp cổ điển, không giảm theo SNR**, do
   các đường NLOS coherent làm méo vector lái. Kết quả 0.017° của bài báo thấp hơn sàn này ~30 lần —
   có thể giải thích bằng việc profile kênh cố định (giả thuyết, chưa kiểm chứng).
4. **Không tạo được dataset 8–10 anten từ tập này**, nhưng nhờ metadata có thể sinh lại tập mới
   dùng đúng kênh của từng mẫu gốc, chỉ đổi hình học mảng.

---

## 1. Nguồn và phương pháp

| Nguồn | Dùng để |
|---|---|
| Bài báo (mục II, IV-A) | Mô hình tín hiệu ct. (1)–(4), tham số công bố |
| `MoDANet-dataset/` (450,120 file `.mat`) và bản convert `data_processed/x_float16.npy` | Đo trực tiếp |
| Repo tác giả [doans/MoDANet](https://github.com/doans/MoDANet) (push 2021-11-29) | Code train MATLAB + `myData_info.mat` |
| [IEEE DataPort](https://ieee-dataport.org/documents/multi-task-learning-dataset-automatic-modulation-classification-and-doa-estimation) | Chỉ chép lại mô tả bài báo, không có code |

Các phép đo là script khảo sát chạy một lần (không nằm trong repo):

- **Probe DOA:** 8,000 mẫu ngẫu nhiên (seed 0, lấy trên toàn dataset, không riêng tập test). Beamscan
  và MUSIC (giả định 1 nguồn), lưới 0.1° trên [−90°, 90°], làm tròn về lớp 1°.
- **Probe phổ:** anten 0, SNR 20 dB, 30 mẫu/điều chế.
- **Đối chiếu metadata:** 3,000 mẫu ngẫu nhiên, giữ SNR ≥ 15 dB; so vector riêng chính của ma trận
  hiệp phương sai với chữ ký dự đoán từ metadata.

---

## 2. Cấu trúc dataset

### 2.1. File và nhãn

| Mục | Giá trị |
|---|---|
| Cây thư mục | `SNR±xxdB/<MOD>/<±DOA>/dataXXXXXXXX.mat` (gốc của tác giả: `dataset_revised\...`) |
| Nội dung file | Duy nhất biến `data`, `single`, kích thước **1024 × 5 × 2** (N × M × I/Q) |
| Header | `MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Sep 21 2021` |
| Số mẫu | 31 SNR × 12 điều chế × 121 DOA × 10 realization = **450,120** |
| Chỉ số file (1-based) | `index = ((snr_i·12 + mod_i)·121 + doa_i)·10 + rep`, với `snr_i = SNR+10`, `doa_i = DOA+60`, `rep = 1..10` |
| Thứ tự `mod_i` | `16QAM, 64QAM, PSK, QPSK, 8PSK, QFSK, 8FSK, 16APSK, 4PAM, LFM, DSB-SC, SSB-SC` — **không** theo thứ tự bài báo liệt kê (đã kiểm bằng tên thư mục, cumulant C42 và metadata tác giả) |
| Cân bằng | Tuyệt đối: 3,720 mẫu/lớp DOA, 37,510 mẫu/lớp điều chế |

Nhãn DOA đọc được theo ba cách tương đương:
`doa = ((index − 1) // 10) % 121 − 60`, tên thư mục `<±DOA>`, hoặc `labels.npy[:, COL_DOA] − 60`.

### 2.2. Thang đo

- Công suất frame ≈ **1 + 10^(−SNR/10)**: tín hiệu (sau kênh) có công suất đơn vị rồi mới cộng nhiễu.
  Ở −10 dB công suất ≈ 10.9; |x| max = 13.44 (4.81 nếu SNR ≥ 0).
- Không có giá trị NaN/Inf. Convert sang fp16: sai số tương đối tối đa 4.9·10⁻⁴.

---

## 3. Dataset thực sự được sinh ra thế nào

### 3.1. Metadata của tác giả

`myData_info.mat` chứa struct `T` 450,120 dòng:

| Trường | Nội dung |
|---|---|
| `Files`, `modulation`, `DOA`, `SNR` | Đường dẫn và nhãn — khớp 450,120/450,120 với dữ liệu |
| `attenuations`, `delays`, `angles` | Tham số từng đường truyền của từng mẫu; `angles(1)` = DOA nhãn (LOS) |
| `Fc` | 100 MHz → λ = 3 m |
| `antennaArray` | `[0 1.5 3 4.5 6]` m → ULA 5 phần tử, d = λ/2 |
| `channel` | MATLAB `string` (scipy không đọc được) |

### 3.2. Quy trình dựng lại

```
for SNR = -10:20
  for mod in [16QAM, 64QAM, PSK, QPSK, 8PSK, QFSK, 8FSK, 16APSK, 4PAM, LFM, DSB-SC, SSB-SC]
    for DOA = -60:60
      for rep = 1:10
        s(t)   = tín hiệu điều chế, N = 1024 mẫu phức
        kênh   = PedA           nếu rep = 1..5
                 VehA rút gọn   nếu rep = 6..10
        θ      = [DOA, (P−1) góc NLOS ~ U(-60°, 60°), bốc lại cho mỗi mẫu]
        x_m(t) = Σ_p g_p · s(t − τ_p) · e^(−j2π·Fc·τ_p) · e^(−j2π·pos_m/λ·sin θ_p)
        chuẩn hóa công suất 1, cộng AWGN theo SNR
        lưu single(1024×5×2)
```

| Profile | Suy hao (dB) | Trễ (ns) | Số mẫu |
|---|---|---|---|
| PedA | 0, −9.7, −19.2, −22.8 | 0, 110, 190, 410 | 225,060 (rep 1–5) |
| VehA **thiếu đường −1 dB / 310 ns** | 0, −9, −10, −15, −20 | 0, 710, 1090, 1730, 2510 | 225,060 (rep 6–10) |

Chi tiết:

- **Hệ số suy hao lưu dạng 10^(dB/10) nhưng được nhân vào biên độ**, nên đường NLOS yếu hơn profile
  chuẩn (−9.7 dB công suất thành −19.4 dB). Bằng chứng ở mục 3.3.
- Mọi `Fc·τ_p` là số nguyên (vd. 100 MHz × 110 ns = 11) → `e^(−j2πFcτ) = 1`: các đường cộng đồng pha.
- Ma trận hiệp phương sai gần hạng 1 (λ₂/λ₁ trung vị 4.6·10⁻³ ở SNR ≥ 15 dB, cỡ mức nhiễu) → trễ
  nhỏ hơn nhiều so với chu kỳ lấy mẫu, các đường NLOS coherent với LOS.

### 3.3. Kiểm chứng metadata ↔ dữ liệu

Độ khớp |⟨chữ ký dự đoán, vector riêng đo⟩| trên 3,000 mẫu SNR ≥ 15 dB:

| Mô hình dự đoán | Trung vị | Phân vị 5% |
|---|---|---|
| Chỉ LOS | 0.99385 | 0.98002 |
| Biên độ = √(hệ số) | 0.97058 | 0.93117 |
| **Biên độ = hệ số** | **0.999993** | **0.999977** |
| Đối chứng: metadata của mẫu kế bên | 0.94244 | 0.85582 |

### 3.4. So với bài báo

| Bài báo (mục IV-A) | Thực tế |
|---|---|
| 1–10 đường NLOS ngẫu nhiên | Cố định 3 (PedA) hoặc 4 (VehA rút gọn) |
| Suy hao ngẫu nhiên [−50, −1] dB | 2 profile cố định |
| Trễ ngẫu nhiên [1, 3000] ns | 2 bộ trễ cố định |
| VehA 6 đường {0, −1, −9, −10, −15, −20} dB | Thiếu đường −1 dB / 310 ns |
| Không nêu góc NLOS | Đều trong [−60°, 60°] |

---

## 4. Đặc trưng dạng sóng (suy ngược, chưa có trong metadata)

Anten 0, SNR 20 dB; tần số tính theo fs:

| Điều chế | Băng thông chiếm dụng 99.5% | Trọng tâm phổ | Đỉnh phổ z⁴ |
|---|---|---|---|
| 16QAM / 64QAM | 0.75 / 0.78 | −0.096 | −0.400 |
| PSK / QPSK / 8PSK | 0.72 / 0.73 / 0.72 | −0.097 | −0.400 |
| 16APSK | 0.76 | −0.096 | −0.398 |
| QFSK / 8FSK | 0.12 / 0.16 | −0.099 | −0.400 |
| 4PAM | 0.87 | −0.001 | −0.004 |
| LFM | 0.10 | +0.001 | +0.002 |
| DSB-SC | 0.39 | −0.098 | −0.400 |
| SSB-SC | 0.11 | −0.100 | −0.404 |

- Mọi điều chế trừ 4PAM và LFM có **lệch tần ≈ −0.10·fs** — bài báo không nhắc.
- BW ≈ 0.75·fs của PSK/QAM phù hợp với 2 mẫu/symbol và roll-off 0.5 (giả thuyết).
- Chưa biết: fs tuyệt đối, bộ lọc tạo dạng xung, độ lệch tần FSK, khoảng quét LFM, tín hiệu tin của
  DSB/SSB.

---

## 5. Khả năng phân tách nhãn DOA (bỏ qua điều chế)

### 5.1. Lý thuyết

- Thông tin DOA nằm trong **pha tương đối giữa các anten** `e^(−jπ·m·sin θ)`, không phụ thuộc dạng sóng.
- Khoảng cách pha giữa 2 lớp kề nhau trên mỗi phần tử: ≈ 0.055 rad tại 0°, chỉ ≈ 0.028 rad tại ±60°.
  Toàn khẩu độ 5 phần tử chỉ tạo khoảng 0.22 rad chênh lệch giữa 2 lớp kề nhau → lưới 1° là rất mịn.

### 5.2. Kết quả MUSIC / beamscan (8,000 mẫu)

Hai phương pháp cho kết quả gần như trùng nhau vì chỉ có 1 nguồn và R gần hạng 1.

| SNR | RMSE | Đúng lớp | Sai tối đa ±1 lớp |
|---|---|---|---|
| −10…−6 dB | 0.98° | 43.3% | 89.6% |
| −5…−1 dB | 0.64° | 57.9% | 97.9% |
| 0…4 dB | 0.55° | 65.4% | 98.5% |
| 5…9 dB | 0.54° | 68.1% | 99.1% |
| 10…14 dB | 0.54° | 65.8% | 98.7% |
| 15…20 dB | **0.52°** | **67.1%** | 99.4% |

Theo điều chế (SNR ≥ 10 dB, % đúng lớp): 16QAM 65.0, 64QAM 68.1, PSK 64.5, QPSK 65.5, 8PSK 63.7,
QFSK 64.3, 8FSK 67.5, 16APSK 69.9, 4PAM 67.3, LFM 64.8, DSB-SC 67.7, SSB-SC 69.3.

Theo vùng góc (SNR ≥ 10 dB):

| \|θ\| | RMSE | Đúng lớp |
|---|---|---|
| 0–29° | 0.45° | 70.9% |
| 30–49° | 0.51° | 68.5% |
| 50–60° | 0.74° | 51.1% |

Kiểm tra mô hình mảng: khớp `sin(ước lượng) = a·sin(thật) + b` cho a = 0.9996, b ≈ 0 → xác nhận
d = λ/2 và dấu của ct. (3).

Ở mức 10 dB, RMSE 0.54° nhất quán với Fig. 6a của bài: MUSIC tốt hơn Network-2 (0.77°).

### 5.3. Nguồn gốc sai số sàn

- Sai số **không giảm theo SNR** và **cố định trong suốt 1024 mẫu** (pha tương đối giữa anten 0 và 1
  dao động ±0.01 rad qua 8 cửa sổ 128 mẫu) → không phải nhiễu.
- Biên độ lệch giữa các anten: hệ số biến thiên trung vị 8.7% (p95 16%); pha lệch ~0.1 rad so với
  vector lái lý tưởng.
- Nguyên nhân (đã xác nhận bằng metadata): tổng coherent của LOS và 3–4 đường NLOS với góc ngẫu nhiên.
  Mức méo này lớn hơn khoảng cách giữa 2 lớp kề nhau → các lớp chồng lấn nếu chỉ giả định 1 nguồn.

### 5.4. Kiểm tra rò rỉ dữ liệu

Cùng (điều chế, DOA, rep) ở các SNR khác nhau **không** dùng chung realization kênh: độ giống chữ ký
không gian ngang với giữa hai rep khác nhau, và metadata cho thấy góc NLOS khác nhau. Không phát hiện
rò rỉ giữa các mức SNR hay giữa các điều chế.

### 5.5. Câu hỏi mở: kết quả 0.017° của bài báo

Bài báo đạt RMSE 0.017° ở 10 dB (gần 100% đúng lớp) — thấp hơn sàn của MUSIC ~30 lần.

**Giả thuyết (chưa kiểm chứng):** vì chỉ có 2 profile biên độ cố định và biết trước, chữ ký
`Σ g_p·a(θ_p)` có 5 số phức (9 bậc tự do thực sau khi bỏ pha chung) nhưng chỉ 4–5 góc chưa biết → hệ
đủ ràng buộc để khôi phục θ₀. Mạng nơ-ron có thể học ngầm cấu trúc này. Cách kiểm: bộ ước lượng bình
phương tối thiểu phi tuyến (NLS) biết trước profile. Nếu đúng, năng lực này **gắn với đặc điểm riêng
của dataset** và khó chuyển sang kênh thực.

---

## 6. Yêu cầu khi đưa dữ liệu vào mô hình khác (chỉ nhãn DOA)

### 6.1. Nhãn

- Chỉ đọc DOA (mục 2.1). Không cần giải mã điều chế → tránh lỗi thứ tự điều chế.
- Vẫn giữ SNR để **đánh giá** (RMSE theo SNR), dù không dùng khi train.

### 6.2. Tiền xử lý — bảo toàn pha và biên độ tương đối giữa anten

| Được phép | Không được |
|---|---|
| Nhân **cùng một** hệ số phức cho cả 5 anten (chuẩn hóa RMS chung của frame, xoay pha chung) | Chuẩn hóa riêng từng anten / từng kênh (per-channel standardization, min-max, LayerNorm theo kênh) |
| Cắt cửa sổ thời gian | Đổi thứ tự anten (trừ phép đảo ở 6.4) |
| | Tách rời I và Q của một anten khỏi nhau |

Công suất frame lộ SNR (mục 2.2); chuẩn hóa theo RMS chung là an toàn cho DOA và nên làm với mô hình
không có BatchNorm ở đầu vào.

### 6.3. Định dạng đầu vào theo họ mô hình

| Họ mô hình | Đầu vào | Ghi chú |
|---|---|---|
| CNN 1D / RNN / Transformer | (1024, 10) hoặc (10, 1024) | 10 kênh = I/Q × 5 anten, ghi rõ ánh xạ kênh |
| CNN 2D / ViT | (2, 5, 1024) hoặc (5, 1024, 2) | Trục anten chỉ dài 5, cẩn thận kernel/padding |
| MLP / mô hình nhẹ | Hiệp phương sai R = XXᴴ/N chuẩn hóa theo trace → 25 số thực (tam giác trên) | Gần như loại bỏ thông tin điều chế; mất thông tin thời gian |
| Đặc trưng thủ công | cos/sin của arg(z_m·z₀*) + tỉ số biên độ | Tránh unwrap pha |
| Mô hình cần N nhỏ | Cửa sổ con ngẫu nhiên | DOA không đổi trong frame; SNR hiệu dụng giảm |

Layout gốc: MATLAB đọc 1024×5×2 rồi permute thành 1024×2×5 (N×IQ×M); memmap của dự án là (M, N, IQ).

### 6.4. Tăng cường dữ liệu hợp lệ

- Xoay pha chung `e^(jφ)` cho cả 5 anten — nhãn không đổi.
- **Đảo thứ tự anten 0…4 → 4…0** — tương đương θ → −θ (sai khác một pha chung). Nhân đôi dữ liệu.
- Dịch vòng thời gian, cắt cửa sổ, scale biên độ chung — nhãn không đổi.

### 6.5. Đầu ra và loss

- Cross-entropy 121 lớp coi sai 1° như sai 100°. Cân nhắc nhãn mềm Gaussian quanh góc đúng, hoặc thêm
  head hồi quy sin θ.
- Báo cáo đồng thời: đúng lớp, đúng ±1°, RMSE — ba số này khác nhau đáng kể (mục 5.2).

### 6.6. Chia tập

- Dùng lại `data_processed/splits/random_seed42.npz` để so với MoDANet.
- Thêm **leave-one-modulation-out** (train 11, test 1) — phép thử duy nhất chứng minh "bất kể điều chế".
- Có thể thêm chia theo profile kênh (train PedA, test VehA) để đo khả năng tổng quát theo kênh.

---

## 7. Ưu điểm và hạn chế của hướng DOA-only

**Ưu điểm**

- Có cơ sở vật lý: DOA nằm trong chữ ký không gian; dữ liệu xác nhận độ chính xác phẳng theo điều chế.
- Pipeline đơn giản: 1 nhãn, 1 loss, không cân bằng trọng số task, không phụ thuộc bảng thứ tự điều chế.
- 3,720 mẫu/lớp, cân bằng tuyệt đối.
- Mở ra mô hình rất nhẹ (25 số thực thay vì 10,240), phù hợp thiết bị hạn chế tài nguyên.
- Dùng được tăng cường theo vật lý và so sánh trực tiếp với MUSIC.

**Hạn chế**

- Mất lợi ích MTL mà bài báo đề xuất; cần thí nghiệm riêng để biết MTL có giúp DOA không.
- Sai số sàn ~0.5° với phương pháp giả định 1 nguồn; nếu mô hình học vượt sàn nhờ profile kênh cố
  định thì khó tổng quát ra kênh thực (mục 5.5).
- Lưới 1° và giới hạn ±60°: góc lệch lưới/ngoài khoảng bị ép về lớp gần nhất; vùng biên kém nhất.
- Chỉ 1 nguồn phát; gắn chặt với ULA 5 phần tử, d = λ/2, băng hẹp.
- "Bất kể điều chế" mới chứng minh trong 12 điều chế; chưa kiểm OFDM/băng rộng.
- Mẫu −10 dB (43% đúng lớp) được đối xử như mẫu sạch nếu bỏ SNR khi train.
- Toàn bộ là mô phỏng: không có lỗi hiệu chuẩn thật, coupling giữa anten.

---

## 8. Tái tạo dataset và mở rộng số anten

### 8.1. Mức tái tạo khả thi

| Thành phần | Trạng thái |
|---|---|
| Kênh từng mẫu (góc, suy hao, trễ), Fc, hình học mảng, nhãn, thứ tự vòng lặp, chuẩn hóa + AWGN | **Biết chính xác** (metadata + đo) |
| fs, số mẫu/symbol, bộ lọc xung, lệch tần, tham số FSK/LFM/DSB/SSB | Phải suy ngược (mục 4) |
| Chuỗi symbol, mẫu nhiễu cụ thể | **Không tái tạo được** — không có seed, repo không có script sinh |

→ Chỉ tái tạo được **tương đương thống kê**, không giống từng bit.

### 8.2. Dataset 8–10 anten

- Không thể lấy từ tập hiện có: tín hiệu tại anten thứ 6–10 không tồn tại trong file. Ngoại suy pha cần
  biết θ (rò rỉ nhãn), và méo đa đường không ngoại suy được nếu không có metadata.
- Chiều ngược lại hợp lệ: bỏ bớt anten để có mảng con 2–4 phần tử liên tiếp.
- **Hướng khuyến nghị:** viết bộ sinh với M là tham số, dùng lại **đúng góc NLOS và profile kênh của
  từng mẫu** trong `myData_info.mat`, chỉ đổi `antennaArray`. Sinh cả M = 5 bằng cùng bộ sinh để so
  sánh công bằng giữa các M.

### 8.3. Kiểm định tập sinh lại (M = 5) với tập gốc

- Thống kê theo lớp: phổ/băng thông, lệch tần, C42, công suất theo SNR, biên độ tối đa.
- Chỉ số DOA: RMSE sàn MUSIC 0.52°, 67% đúng lớp, CV biên độ anten 8.7%, λ₂/λ₁.
- Chữ ký không gian dự đoán từ metadata khớp ≥ 0.9999.
- Chéo mô hình: train trên tập sinh lại → test trên tập gốc, và ngược lại.

---

## 9. Code train của tác giả (ghi nhận, không áp dụng)

`TrainMIMO_network.m` khác cả bài báo lẫn MODANET_INSTRUCTION.md:

| Mục | Code tác giả |
|---|---|
| Batch | 256 |
| Filter | 9×1 |
| Learning rate | 0.01 / (1 + 0.01·iteration), SGDM momentum 0.9 |
| Epoch | vòng `for epoch = 3:10` (có vẻ chạy tiếp từ checkpoint) |
| Chia tập | `randperm` không đặt seed, 80/20; tập validation dùng luôn làm tập test |
| Đọc dữ liệu | `load(...,'data')` rồi `permute([1 3 2 4])` → 1024×2×5×B |

---

## 10. Việc tiếp theo đề xuất

1. Lưu `myData_info.mat` của tác giả vào dự án (hiện chỉ có bản tạm) — nền cho mọi bước dưới.
2. Chạy MUSIC trên đúng tập test `random_seed42`, đủ 31 mức SNR, làm baseline so với MoDANet.
3. Kiểm giả thuyết mục 5.5 bằng NLS biết trước profile kênh.
4. Đánh giá DOA tách theo profile kênh (PedA vs VehA rút gọn).
5. Suy ngược đầy đủ tham số dạng sóng (mục 4) → bộ sinh dữ liệu M anten.
