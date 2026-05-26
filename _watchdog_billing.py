"""Throwaway — watchdog that swaps the project's billing account from
the Education-credit account to 'My Billing Account 2' once the scan
has spent ~$33 (leaving ~$5 unspent on the Education credit).

Polls poi_scan_cos0.75.jsonl every 5 min. The main vlm scan script is
unaware; gcloud just re-targets which billing account new Vertex calls
bill, and the running script continues uninterrupted.
"""

import subprocess, sys, time
from pathlib import Path

OUT_FILE   = Path("data/cities/zurich/poi_scan_cos0.75.jsonl")
THRESH_ROWS = 2832                                  # 82 copies + 2750 scans = ~$33
NEW_ACCOUNT = "01DF8D-AAE976-C395FE"               # My Billing Account 2
PROJECT     = "cs231n-navlm-2026"
POLL_S      = 300                                    # check every 5 min


def now():
    return time.strftime("%H:%M:%S")


def count_rows():
    if not OUT_FILE.exists():
        return 0
    return sum(1 for _ in OUT_FILE.open(encoding="utf-8"))


def gcloud_switch():
    cmd = ("gcloud billing projects link "
           f"{PROJECT} --billing-account={NEW_ACCOUNT}")
    print(f"[{now()}][watchdog] running:  {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"[{now()}][watchdog] returncode={r.returncode}", flush=True)
    if r.stdout:
        print(f"  stdout: {r.stdout.strip()}", flush=True)
    if r.stderr:
        print(f"  stderr: {r.stderr.strip()}", flush=True)
    return r.returncode == 0


print(f"[{now()}][watchdog] starting. Will switch billing -> "
      f"{NEW_ACCOUNT} once {OUT_FILE.name} reaches {THRESH_ROWS} rows.",
      flush=True)

while True:
    n = count_rows()
    # ~2750 of these are paid scans (the first 82 were free copies)
    paid_scans = max(0, n - 82)
    cost = paid_scans * 0.012
    remaining_credit = 38 - cost
    if n >= THRESH_ROWS:
        print(f"[{now()}][watchdog] threshold hit "
              f"({n} >= {THRESH_ROWS}). cost ~${cost:.2f}, "
              f"~${remaining_credit:.2f} left on Education credit.",
              flush=True)
        if gcloud_switch():
            print(f"[{now()}][watchdog] BILLING SWITCHED. "
                  f"Main scan continues uninterrupted on new account.",
                  flush=True)
            sys.exit(0)
        else:
            print(f"[{now()}][watchdog] switch FAILED -- will retry "
                  f"in {POLL_S}s", flush=True)
    else:
        eta_h = (THRESH_ROWS - n) * 13 / 3600       # ~13 s/scan
        print(f"[{now()}][watchdog] rows={n}/{THRESH_ROWS}  "
              f"paid_scans={paid_scans}  cost~${cost:.2f}  "
              f"edu_remaining~${remaining_credit:.2f}  "
              f"eta~{eta_h:.1f}h", flush=True)
    time.sleep(POLL_S)
