#!/usr/bin/env bash
# Train đầy đủ MoDANet ở chế độ nền, log ra file. Tắt terminal / mất SSH vẫn chạy.
# Chạy lại đúng lệnh này sau khi mất điện / máy bị ngắt -> tự --resume từ last.pt
# (kể cả khi chuyển sang máy khác: micro-batch tự tính lại theo GPU mới, batch hiệu dụng giữ 128).
#
#   bash run_train.sh                       # preset paper; micro-batch theo VRAM (2 GB -> 32×4, 4 GB -> 64×2, ≥6 GB -> 128×1)
#   bash run_train.sh --in-memory           # server nhiều RAM (≥ 12 GB trống): nạp dữ liệu vào RAM, nhanh nhất
#   BS=64 ACC=2 bash run_train.sh           # tự đặt micro-batch
#   OUT=outputs/run02 bash run_train.sh --split-mode group
#   bash run_train.sh --max-hours 8         # dừng an toàn sau ~8 giờ, chạy lại để tiếp
#   PY=/path/to/python bash run_train.sh    # chỉ định python
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
  for c in ../.venv/bin/python .venv/bin/python "$(command -v python3 || true)" "$(command -v python || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then PY="$c"; break; fi
  done
fi
OUT=${OUT:-outputs/run01}
mkdir -p "$OUT"
if [ -f "$OUT/last.pt" ]; then
  ARGS=(--resume "$OUT/last.pt")
  echo "Tìm thấy $OUT/last.pt -> chạy tiếp"
else
  ARGS=(--preset paper --out "$OUT")
fi
[ -n "${BS:-}" ] && ARGS+=(--batch-size "$BS")
[ -n "${ACC:-}" ] && ARGS+=(--accum-steps "$ACC")
nohup "$PY" -u scripts/train.py "${ARGS[@]}" "$@" >> "$OUT/train.out" 2>&1 &
echo "PID $!  |  python: $PY  |  theo dõi: tail -f $OUT/train.out  |  dừng: kill $!"
