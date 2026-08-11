#!/usr/bin/env bash
# Stage runner. Thin wrapper over `python -m src.cli` — the logic lives there,
# this file exists so a rented box needs one command and no argument archaeology.
#
#   ./run.sh test                    unit + smoke, no GPU, seconds
#   ./run.sh stub    r0              whole slice on the canned model, no GPU
#   ./run.sh pilot   r0              REAL model, 2 prompts, ~3 min — run before renting a night
#   ./run.sh constitute r0           model writes its own criteria from your labels
#   ./run.sh sample  r0              drafts + prefills + self-review revisions
#   ./run.sh pair    r0              pick which pairs to judge      (no GPU)
#   ./run.sh spread  r0              per-variant signal report      (no GPU)
#   ./run.sh pool    r1              constitute -> sample -> pair -> judge -> spread
#   ./run.sh judge   r0              judge them                 (needs a model)
#   ./run.sh validate r0             audit the judge — THE GATE
#   ./run.sh peek    r0              read a run, safe mid-experiment
#   ./run.sh label   r0              hand-label pairs, blind to the judge
#   ./run.sh night   r0              full sweep under nohup, pushes to the Hub
#   ./run.sh night   r0 --kill       ... and destroys the vast instance when done
#
# Recommended first-time order (label BEFORE judging — see README):
#   ./run.sh pilot p0 ; ./run.sh pairs r0 --push ; ./run.sh label r0 ; ./run.sh judge r0
#   ./run.sh all     r0              pairs -> judge -> validate, stopping on failure
#   ./run.sh push    r0              mirror runs/r0 to the logs dataset repo
#   ./run.sh pull    r0              fetch runs/r0 back from the Hub
#   ./run.sh whoami                  check HF_TOKEN works, before renting a box
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
    $PY -m src.cli sample --run "$RUN" --stub --fresh --limit 4
    $PY -m src.cli pair   --run "$RUN"
    $PY -m src.cli judge  --run "$RUN" --stub --fresh
    # validate is expected to FAIL here: the stub judge has planted position
    # bias. `|| true` keeps the demo going so you can read the report.
    $PY -m src.cli validate --run "$RUN" || true
    $PY -m src.cli peek --run "$RUN"
    ;;

  pilot|label|judge|validate|peek|constitute|sample|pair|spread)
    need_run
    $PY -m src.cli "$CMD" --run "$RUN" "$@"
    ;;

  push)   need_run; $PY -m src.cli sync push-run --run "$RUN" "$@" ;;
  pull)   need_run; $PY -m src.cli sync pull-run --run "$RUN" "$@" ;;
  whoami) $PY -m src.cli sync whoami --run _ ;;

  pool)
    # The full variant experiment. constitute -> sample -> pair -> judge -> spread.
    # `pair` and `spread` need no GPU; they are in the chain so one command
    # produces a readable answer rather than four files to join by hand.
    need_run
    # Only flags EVERY stage understands may be forwarded. `-n` means something
    # to constitute and nothing to sample; `--limit` is the reverse. Forwarding
    # blindly would fail three stages in, after the expensive one had run.
    for arg in "$@"; do
      case "$arg" in
        --stub|--push|--fresh) ;;
        *) echo "./run.sh pool forwards only --stub --push --fresh (got '$arg')."
           echo "For -n / --limit / --domain, run the stages individually."
           exit 2 ;;
      esac
    done
    export PYTHONUNBUFFERED=1
    $PY -m src.cli constitute --run "$RUN" "$@"
    $PY -m src.cli sample     --run "$RUN" "$@"
    $PY -m src.cli pair       --run "$RUN"
    $PY -m src.cli judge      --run "$RUN" "$@"
    $PY -m src.cli spread     --run "$RUN"
    ;;

  night)
    need_run
    KILL=0
    for arg in "$@"; do [ "$arg" = "--kill" ] && KILL=1; done

    # Detached, unbuffered, pushing on each stage's success path. nohup because
    # an ssh drop must not kill a paid-for run; PYTHONUNBUFFERED because python
    # block-buffers stdout when it is a file, so `tail -f` would show nothing
    # for hours. tmux belongs on the BOX, not on your laptop.
    export PYTHONUNBUFFERED=1

    # Chain shape matters. `pairs && judge` means a destroy can only ever happen
    # after both have completed AND pushed. `validate` is wrapped in `|| true`
    # because a judge failing its gates is a RESULT, not a crash -- it must not
    # keep the box alive at $/hr, and it must not prevent the log being saved.
    # Destroy must NEVER block on a prompt: this runs detached, with no one at
    # the keyboard, so a confirmation question means the box just stays alive
    # and bills. `vastai destroy` asks "[y/N]" by default, so `y` is piped in.
    # The curl path is the fallback and is prompt-free by construction; it needs
    # VAST_API_KEY (put it in the same .env as HF_TOKEN).
    KILLCMD=""
    if [ "$KILL" = 1 ]; then
      KILLCMD="
      cp '$RUN.log' 'runs/$RUN/console.log' 2>/dev/null || true
      $PY -m src.cli sync push-run --run '$RUN' --message '$RUN: console log' || true
      ID=\"\${CONTAINER_ID:-}\"
      if [ -z \"\$ID\" ]; then
        echo 'NOT DESTROYED: CONTAINER_ID unset (not a vast box?). Destroy it yourself.'
      elif [ -n \"\${VAST_API_KEY:-}\" ]; then
        echo \"destroying instance \$ID via API\"
        curl -s -X DELETE \"https://console.vast.ai/api/v0/instances/\$ID/?api_key=\$VAST_API_KEY\"
      elif command -v vastai >/dev/null; then
        echo \"destroying instance \$ID via CLI\"
        printf 'y\\n' | vastai destroy instance \"\$ID\"
      else
        echo 'NOT DESTROYED: no VAST_API_KEY and no vastai CLI. Destroy it yourself.'
      fi"
    fi

    nohup bash -c "\
      $PY -m src.cli constitute --run '$RUN' --push && \
      $PY -m src.cli sample     --run '$RUN' --push && \
      $PY -m src.cli pair       --run '$RUN' && \
      $PY -m src.cli judge      --run '$RUN' --push && \
      $PY -m src.cli spread     --run '$RUN' && \
      { $PY -m src.cli validate --run '$RUN' --push || true; } \
      $KILLCMD" > "$RUN.log" 2>&1 &
    echo "started pid $! -> $RUN.log"
    if [ "$KILL" = 1 ]; then
      echo "will destroy the instance when the chain finishes"
    fi
    echo "watch with:  tail -f $RUN.log"
    ;;

  all)
    need_run
    $PY -m src.cli sample --run "$RUN" "$@"
    $PY -m src.cli pair   --run "$RUN"
    $PY -m src.cli judge  --run "$RUN" "$@"
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
