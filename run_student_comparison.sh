#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
comparison_python="$project_root/work/metal-env/bin/python"
report_python="$project_root/work/venv/bin/python"

if [[ ! -x "$comparison_python" ]]; then
    echo 'Missing GPU environment. See MODEL_COMPARISON.md for setup.' >&2
    exit 1
fi

cpu_pattern='work/mlx-env/bin/python compare_students[.]py --candidate qwen3 --weights work/hf/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit'
pkill -TERM -f "$cpu_pattern" || true
for attempt in 1 2 3 4 5; do
    if ! pgrep -f "$cpu_pattern" >/dev/null; then
        break
    fi
    sleep 1
done
if pgrep -f "$cpu_pattern" >/dev/null; then
    echo 'The previous CPU attempt is still active. Stop it before retrying.' >&2
    exit 1
fi

"$comparison_python" - <<'PY'
import json
from pathlib import Path

path = Path('amc_results/model_comparison/qwen3.json')
if path.exists():
    document = json.loads(path.read_text())
    if 'cpu' in document.get('device', ''):
        if not document['complete']:
            document.update(status='stopped_before_gpu_retry', complete=False)
        path.with_name('qwen3_cpu_attempt.json').write_text(json.dumps(document, indent=2) + '\n')
PY

"$comparison_python" -c 'import mlx.core as mx; assert mx.metal.is_available(), "Run from a normal Terminal with access to the Mac GPU"'

export HF_HOME="$project_root/work/hf"
while IFS= read -r candidate; do
    "$comparison_python" compare_students.py --candidate "$candidate"
done < <("$comparison_python" -c 'import json; print("\n".join(json.load(open("comparison_config.json"))["candidates"]))')
"$report_python" comparison_report.py
echo 'Comparison finished. Saved results are ready for review and publication.'
