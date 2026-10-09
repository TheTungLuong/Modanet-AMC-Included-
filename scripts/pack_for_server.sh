#!/usr/bin/env bash
# Đóng gói CODE (không gồm dữ liệu 9.2 GB) để đưa lên server:
#   bash scripts/pack_for_server.sh            -> dist/modanet_code.tar.gz (~1 MB, gồm 22 file .mat mẫu cho test)
# Sau đó xem README mục 6 để upload code + dữ liệu lên vast.ai.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
tar --exclude='./data_processed' --exclude='./outputs' --exclude='./dist' \
    --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pt' --exclude='*.tmp' \
    -czf dist/modanet_code.tar.gz --transform 's,^\.,modanet,' .
ls -lh dist/modanet_code.tar.gz
echo "Nội dung: $(tar -tzf dist/modanet_code.tar.gz | wc -l) file"
[ -f data_processed/SHA256SUMS ] || echo "⚠️  Chưa có data_processed/SHA256SUMS (tạo: cd data_processed && sha256sum x_float16.npy labels.npy meta.json splits/*.npz > SHA256SUMS)"
