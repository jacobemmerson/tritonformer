#!/usr/bin/env bash
# Lock clocks before benchmarking. Requires root.
# Pick a graphics clock the card can sustain thermally -- for a laptop
# 1650 Ti that is well below boost. Check supported values with:
#   nvidia-smi --query-supported-clocks=gr --format=csv
set -euo pipefail
CLOCK="${1:-1200}"
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc "${CLOCK},${CLOCK}"
echo "locked graphics clock to ${CLOCK} MHz; reset with: sudo nvidia-smi -rgc"
