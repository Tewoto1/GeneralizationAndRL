#!/usr/bin/env bash
# Stage runner. Thin wrapper over `python -m src.cli` — the logic lives there,
# this file exists so a rented box needs one command and no argument archaeology.
#
#   ./run.sh test                    unit + smoke, no GPU, seconds
#   ./run.sh stub    r0              whole slice on the canned model, no GPU
#   ./run.sh pairs   r0              generate answer pairs      (needs a model)
#   ./run.sh judge   r0              judge them                 (needs a model)
#   ./run.sh validate r0             audit the judge — THE GATE
#   ./run.sh peek    r0              read a run, safe mid-experiment
#   ./run.sh all     r0              pairs -> judge -> validate, stopping on failure
#
# Env: MODEL / ADAPTER override configs/model.json. PY overrides the interpreter.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
CMD="${1:-}"; shift || true
RUN="${1:-}"; [ $# -gt 0 ] && shift || true

need_run() { [ -n "$RUN" ] || { echo "usage: ./run.sh $CMD <run-name>"; exit 2; }; }

case "$CMD" in
  test)
    export PYTHONUNBUFFERED=1
    $PY -m pytest tests/ -q "$@"
    ;;

  stub)
    need_run
    $PY -m src.cli pairs --run "$RUN" --stub --fresh
    $PY -m src.cli judge --run "$RUN" --stub --fresh
    # validate is expected to FAIL here: the stub judge has planted position
    # bias. `|| true` keeps the demo going so you can read the report.
    $PY -m src.cli validate --run "$RUN" || true
    $PY -m src.cli peek --run "$RUN"
    ;;

  pairs|judge|validate|peek)
    need_run
    $PY -m src.cli "$CMD" --run "$RUN" "$@"
    ;;

  all)
    need_run
    $PY -m src.cli pairs    --run "$RUN" "$@"
    $PY -m src.cli judge    --run "$RUN" "$@"
    # Deliberately not `|| true`: a judge that fails validation must stop the
    # pipeline. Everything downstream is a function of the judge, so producing
    # data from a failed instrument is worse than producing none.
    $PY -m src.cli validate --run "$RUN"
    $PY -m src.cli peek     --run "$RUN"
    ;;

  ""|-h|--help)
    sed -n '2,20p' "$0"
    ;;

  *)
    echo "unknown stage: $CMD"; sed -n '2,20p' "$0"; exit 2
    ;;
esac
