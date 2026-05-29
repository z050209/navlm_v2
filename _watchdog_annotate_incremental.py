"""Throwaway — fires the incremental annotation pass as soon as the
main annotation batch (bg task b9gc1sxzf) completes.

  current main batch  →  ~5,091 (frame, dest) rows on the OLD 1,697-
                          frame cohort. The script writes them to
                          annotations_strict.jsonl as it goes.
  this watchdog       →  polls the log every 90 s. When it sees
                          "=== annotation summary ===", fires another
                          annotate.py run with the SAME --seed 42 +
                          same output file. Resume logic in annotate.py
                          skips every (video, frame_id, dest_name)
                          already in the file (5,091 of them) and
                          processes only the new 99 frames × 3 dest
                          ≈ 297 fresh calls (~$6.50, ~4 h on Pro/Vertex).

Exits cleanly when the incremental run finishes.
"""

import os, subprocess, sys, time

os.environ["MSYS_NO_PATHCONV"] = "1"
os.environ["PYTHONUTF8"] = "1"

PYTHON = r"C:\Users\z0502\anaconda3\envs\navlm_v2\python.exe"
MAIN_LOG = (r"C:\Users\z0502\AppData\Local\Temp\claude\\"
            r"G--My-Drive-cs231n-project-cs231n\\"
            r"b5bcb9d3-30bb-4725-93c3-a3caf084e779\tasks\b9gc1sxzf.output")
POLL_S = 90
DONE_MARKER = "=== annotation summary ==="


def now():
    return time.strftime("%H:%M:%S")


def main_batch_done():
    try:
        with open(MAIN_LOG, "rb") as f:
            f.seek(-8192, 2)
            tail = f.read().decode("utf-8", errors="replace")
        return DONE_MARKER in tail
    except (FileNotFoundError, OSError):
        return False


print(f"[{now()}][watchdog] polling main-annotation log every {POLL_S}s "
      f"for '{DONE_MARKER}'", flush=True)
while True:
    if main_batch_done():
        print(f"[{now()}][watchdog] main batch done. "
              f"firing incremental annotate.", flush=True)
        break
    print(f"[{now()}][watchdog] not yet — sleeping {POLL_S}s", flush=True)
    time.sleep(POLL_S)

# Fire the incremental run — same args, same output file. annotate.py
# will see the 5,091 already-done rows, skip them, and process only the
# new famous-POI frames in the expanded trusted_frames.jsonl.
cmd = [PYTHON, "-u", "-m", "src.annotate",
       "--limit", "0",
       "--prompt-variant", "strict",
       "--seed", "42"]
print(f"[{now()}][watchdog] running: {' '.join(cmd)}", flush=True)
r = subprocess.run(cmd)
print(f"[{now()}][watchdog] incremental annotate exit {r.returncode}",
      flush=True)
sys.exit(r.returncode)
