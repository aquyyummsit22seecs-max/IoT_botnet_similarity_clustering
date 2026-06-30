"""log_utils.py — Logging helpers for reproducible pipeline execution."""

import os, sys, atexit, traceback, builtins
from datetime import datetime

_STATE: dict = {}
_ORIGINAL_PRINT = builtins.print
_PIPELINE_PREFIXES = ("[", "Pipeline", "Step", "===", "---")


def _is_pipeline_line(text):
    return text.lstrip().startswith(_PIPELINE_PREFIXES)

def _ts_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _append(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def log_kv(title, mapping):
    """Print a titled key-value block (always treated as a pipeline line)."""
    print(title)
    for key, value in mapping.items():
        print(f"  {key}: {value}")

def _install_timestamped_print(shared_log, script_log):
    def ts_print(*args, **kwargs):
        sep   = kwargs.get("sep",  " ")
        end   = kwargs.get("end",  "\n")
        file  = kwargs.get("file", sys.stdout)
        flush = kwargs.get("flush", False)
        text  = sep.join(str(a) for a in args)

        if file not in (sys.stdout, sys.stderr, None):
            _ORIGINAL_PRINT(*args, **kwargs)
            return

        if _is_pipeline_line(text):
            ts_line = f"[{_ts_now()}] {text}"
            _ORIGINAL_PRINT(ts_line, end=end, file=file, flush=flush)
            _append(shared_log, ts_line if end == "\n" else ts_line + end.rstrip("\n"))
            _append(script_log,  ts_line if end == "\n" else ts_line + end.rstrip("\n"))
        else:
            _ORIGINAL_PRINT(text, end=end, file=file, flush=flush)

    builtins.print = ts_print


def setup_script_logging(script_name):
    """Initialise per-script logging. Call once at the top of each pipeline script."""
    os.makedirs("output/logs", exist_ok=True)
    pipeline_start = os.environ.get("PIPELINE_START_TS") or _ts_now()
    shared_log = os.environ.get("PIPELINE_LOG_PATH", "output/logs/pipeline.log")
    script_log = f"output/logs/{os.path.splitext(os.path.basename(script_name))[0]}.log"
    start_ts = _ts_now()

    _STATE.clear()
    _STATE.update({
        "script_name": script_name,
        "shared_log":  shared_log,
        "script_log":  script_log,
        "start_ts":    start_ts,
        "failed":      False,
    })

    line = f"[SCRIPT START] {script_name} | started={start_ts} | pipeline_started={pipeline_start}"
    _append(shared_log, line)
    _append(script_log,  line)
    _install_timestamped_print(shared_log, script_log)

    def _on_exit():
        status = "FAILED" if _STATE.get("failed") else "SUCCESS"
        end_line = f"[SCRIPT END]   {script_name} | ended={_ts_now()} | status={status}"
        _append(shared_log, end_line)
        _append(script_log,  end_line)
        builtins.print = _ORIGINAL_PRINT

    def _hook(exc_type, exc, tb):
        _STATE["failed"] = True
        err_line = (f"[SCRIPT ERROR] {script_name} | ended={_ts_now()} | "
                    f"status=FAILED | error={exc_type.__name__}: {exc}")
        _append(shared_log, err_line)
        _append(script_log,  err_line)
        _append(script_log,  "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)

    atexit.register(_on_exit)
    sys.excepthook = _hook
    return {"shared_log": shared_log, "script_log": script_log}
