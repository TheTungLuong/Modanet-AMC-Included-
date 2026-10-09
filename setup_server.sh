#!/usr/bin/env bash
# Chuẩn bị máy mới / server (vd. vast.ai) — chạy MỘT lần trong thư mục modanet/:
#   bash setup_server.sh
# 1) cài thư viện (giữ torch có sẵn của image), 2) kiểm GPU, 3) kiểm checksum dữ liệu, 4) chạy test.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$(command -v python3 || command -v python)}
echo "== Python: $PY ($($PY --version 2>&1))"

echo "== 1. Thư viện"
if $PY -c "import torch" 2>/dev/null; then
  echo "   torch có sẵn: $($PY -c 'import torch; print(torch.__version__, "CUDA", torch.version.cuda)')"
fi
$PY -m pip install -q -r requirements.txt

echo "== 2. Thiết bị"
$PY -m src.utils.device || echo "   ⚠️  GPU chưa dùng được — xem thông báo ở trên"

echo "== 3. Dữ liệu"
DATA=${MODANET_PROCESSED_DIR:-data_processed}
echo "   thư mục: $DATA"
if [ -f "$DATA/meta.json" ]; then
  if [ -f "$DATA/SHA256SUMS" ]; then
    (cd "$DATA" && sha256sum -c SHA256SUMS) || { echo "❌ checksum lệch -> upload lại dữ liệu"; exit 1; }
  else
    echo "   (không có SHA256SUMS — bỏ qua kiểm checksum)"
  fi
else
  echo "   ⚠️  Chưa có data_processed/ — xem README mục 6 (upload memmap hoặc convert từ .mat)"
fi

echo "== 4. Test (~3 phút)"
$PY -m pytest -q -x
echo "== Xong. Smoke test:  $PY scripts/train.py --smoke --subset 20000 --epochs 2"
echo "        Train đầy đủ: bash run_train.sh --in-memory   (hoặc bỏ --in-memory nếu RAM < 12 GB)"
