#!/usr/bin/env bash
# Concurrent load test for the swellmd daemon.
#
# Usage: ./llmserv/main/features/daemon/scripts/load_test.sh [N] [URL]
#   N   — number of concurrent requests (default: 4)
#   URL — endpoint (default: http://127.0.0.1:8080/v1/chat/completions)
#
# Output lands in $SCRIPT_DIR/out/ (gitignored).
#
# Note on Windows/Cygwin: bash fork() is expensive. N > ~20 may hit
# "Resource temporarily unavailable" due to cygheap fork exhaustion,
# not server issues. For large N, use a Python/Rust client instead.

set -u

N="${1:-4}"
URL="${2:-http://127.0.0.1:8080/v1/chat/completions}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$SCRIPT_DIR/out"

rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

echo "Firing $N concurrent requests at $URL"
START=$(date +%s.%N)

for i in $(seq 1 $N); do
  (
    body="{\"model\":\"gemma\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi request $i, reply briefly.\"}],\"max_tokens\":8,\"temperature\":0.0}"
    t0=$(date +%s.%N)
    http=$(curl -s -o "$OUTDIR/resp_$i.json" -w "%{http_code}" --max-time 600 \
      -X POST "$URL" -H "Content-Type: application/json" -d "$body")
    t1=$(date +%s.%N)
    dt=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
    echo "$i $http $dt" > "$OUTDIR/stat_$i.txt"
  ) &
done

wait
END=$(date +%s.%N)
WALL=$(awk "BEGIN{printf \"%.2f\", $END - $START}")

OK=$(cat "$OUTDIR"/stat_*.txt 2>/dev/null | awk '$2==200{c++} END{print c+0}')
ERR=$(cat "$OUTDIR"/stat_*.txt 2>/dev/null | awk '$2!=200{c++} END{print c+0}')
AVG=$(cat "$OUTDIR"/stat_*.txt 2>/dev/null | awk '{s+=$3; c++} END{if(c>0) printf "%.2f", s/c; else print "0"}')
MAX=$(cat "$OUTDIR"/stat_*.txt 2>/dev/null | awk 'BEGIN{m=0} {if($3>m)m=$3} END{printf "%.2f", m}')
MIN=$(cat "$OUTDIR"/stat_*.txt 2>/dev/null | awk 'BEGIN{m=1e9} {if($3<m && $3>0)m=$3} END{printf "%.2f", m}')

echo ""
echo "=== RESULTS ==="
echo "Total requests: $N"
echo "Successful:     $OK"
echo "Failed:         $ERR"
echo "Wall clock:     ${WALL}s"
echo "Per-request min/avg/max: ${MIN}s / ${AVG}s / ${MAX}s"
echo ""
echo "=== STATUS DISTRIBUTION ==="
cat "$OUTDIR"/stat_*.txt 2>/dev/null | awk '{print $2}' | sort | uniq -c
