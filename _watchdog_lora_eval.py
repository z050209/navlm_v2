"""Throwaway — watchdog that fires the L-given eval cells as soon as
the L-given SFT adapter lands on the navlm-ckpts volume.

Polls every 90 s. When `/lora_given_r16_e2/adapter_model.safetensors`
appears on the volume, fires:
    modal run eval_modal.py --condition L-given --ablation video --no-anchor
    modal run eval_modal.py --condition L-given --ablation poi   --no-anchor

Both go to the same `--run-id` (today's timestamp) so they land in
one directory under `navlm-eval`. Exits cleanly when both eval calls
return.
"""

import datetime
import os
import subprocess
import sys
import time

os.environ["MSYS_NO_PATHCONV"] = "1"
os.environ["PYTHONUTF8"] = "1"

PYTHON = r"C:\Users\z0502\anaconda3\envs\navlm_v2\python.exe"
ADAPTER_PATH_ON_VOLUME = "/lora_given_r16_e2/adapter_model.safetensors"
POLL_S = 90


def now():
    return time.strftime("%H:%M:%S")


def adapter_exists():
    """True iff the trained LoRA file is on the navlm-ckpts volume."""
    r = subprocess.run(
        [PYTHON, "-m", "modal", "volume", "ls", "navlm-ckpts",
         "/lora_given_r16_e2"],
        capture_output=True, text=True, timeout=60)
    return "adapter_model.safetensors" in (r.stdout or "")


def run_eval(ablation, run_id):
    cmd = [PYTHON, "-u", "-m", "modal", "run", "eval_modal.py",
           "--condition", "L-given",
           "--ablation", ablation,
           "--run-id", run_id,
           "--no-anchor"]
    print(f"[{now()}][watchdog] firing: {' '.join(cmd[3:])}", flush=True)
    r = subprocess.run(cmd, capture_output=False)
    print(f"[{now()}][watchdog] eval L-given × {ablation} returned "
          f"exit {r.returncode}", flush=True)


def main():
    print(f"[{now()}][watchdog] starting — polling every {POLL_S} s "
          f"for {ADAPTER_PATH_ON_VOLUME}", flush=True)
    while True:
        if adapter_exists():
            print(f"[{now()}][watchdog] adapter detected — firing eval",
                  flush=True)
            break
        print(f"[{now()}][watchdog] not yet — sleeping {POLL_S} s",
              flush=True)
        time.sleep(POLL_S)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Lgiven"
    print(f"[{now()}][watchdog] run_id = {run_id}", flush=True)
    run_eval("video", run_id)
    run_eval("poi", run_id)
    print(f"[{now()}][watchdog] both L-given cells fired. "
          f"Pull: python pull_eval.py {run_id}", flush=True)


if __name__ == "__main__":
    main()
