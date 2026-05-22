#!/usr/bin/env python3
"""Daily API-spend log for the NavLM project.

Tallies the cost of the API calls this project generates and writes a
per-day breakdown to costs/daily_spend.md. Covers the spend we control:

  - Street View Static API : $0.007 / image  (grid crawl + POI fetch)
  - Gemini API             : per-token, read from costs/gemini_calls.jsonl
  - Mapillary + SV metadata: free (not counted)

NOT covered: GCP VM compute (~$0.13/hr) — see the GCP Billing console.

Run from navlm_ss/:
    python track_spend.py
"""

import datetime
import json
from collections import defaultdict
from pathlib import Path

SV_PRICE = 0.007        # USD per Street View Static image
BUDGET = 50.0           # GCP education credit

SV_DIRS = [Path("data/cities/streetview/zurich/images"),
           Path("../preview/streetview_poi")]
GEMINI_LEDGER = Path("costs/gemini_calls.jsonl")
OUT = Path("costs/daily_spend.md")


def date_of(p):
    return datetime.date.fromtimestamp(p.stat().st_mtime).isoformat()


def main():
    sv_by_date = defaultdict(int)
    for d in SV_DIRS:
        if d.exists():
            for f in d.rglob("*.jpg"):
                sv_by_date[date_of(f)] += 1

    gem_cost = defaultdict(float)
    gem_calls = defaultdict(int)
    if GEMINI_LEDGER.exists():
        for ln in GEMINI_LEDGER.open(encoding="utf-8"):
            r = json.loads(ln)
            d = r["ts"][:10]
            gem_cost[d] += r.get("cost_usd", 0.0)
            gem_calls[d] += 1

    days = sorted(set(sv_by_date) | set(gem_cost))
    rows, cum = [], 0.0
    for d in days:
        sv_n = sv_by_date.get(d, 0)
        sv_c = sv_n * SV_PRICE
        gm_c = gem_cost.get(d, 0.0)
        tot = sv_c + gm_c
        cum += tot
        rows.append((d, sv_n, sv_c, gem_calls.get(d, 0), gm_c, tot, cum))
    total = cum

    lines = [
        "# NavLM — daily API spend", "",
        f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M}_", "",
        "API spend this project generates. Street View Static API "
        "@ $0.007/image; Gemini @ token list price. Mapillary and "
        "Street View *metadata* calls are free. **GCP VM compute "
        "(~$0.13/hr) is NOT included here** — see the Billing console.", "",
        "| Date | SV images | SV $ | Gemini calls | Gemini $ | Day total | Cumulative |",
        "|------|----------:|-----:|-------------:|---------:|----------:|-----------:|",
    ]
    for d, svn, svc, gmn, gmc, tot, c in rows:
        lines.append(f"| {d} | {svn} | ${svc:.2f} | {gmn} | ${gmc:.2f} "
                     f"| ${tot:.2f} | ${c:.2f} |")
    lines += [
        "",
        f"**Total API spend: ${total:.2f}** of the $50 GCP credit  ",
        f"**Remaining (API only): ${BUDGET - total:.2f}**", "",
        "> Whole-account spend (incl. the dev VM) is in the GCP Billing "
        "console: https://console.cloud.google.com/billing", "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    # browsable HTML dashboard
    html_out = Path("../preview/spend.html")
    trs = "".join(
        f"<tr><td>{d}</td><td>{svn}</td><td>${svc:.2f}</td><td>{gmn}</td>"
        f"<td>${gmc:.2f}</td><td><b>${tot:.2f}</b></td><td>${c:.2f}</td></tr>"
        for d, svn, svc, gmn, gmc, tot, c in rows)
    pct = min(100, total / BUDGET * 100)
    bar_color = "#3a3" if pct < 75 else ("#e90" if pct < 100 else "#c00")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>NavLM API spend</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:30px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:right}}
th{{background:#f4f4f4}}td:first-child,th:first-child{{text-align:left}}
.bar{{background:#eee;border-radius:6px;height:26px;overflow:hidden}}
.fill{{background:{bar_color};height:100%;width:{pct:.1f}%;color:#fff;
text-align:right;padding-right:8px;box-sizing:border-box;line-height:26px;font-weight:bold}}
.muted{{color:#888;font-size:13px}}</style></head><body>
<h2>NavLM — API spend</h2>
<div class="bar"><div class="fill">${total:.2f}</div></div>
<p><b>${total:.2f}</b> of $50 GCP credit used &nbsp;·&nbsp; <b>${BUDGET-total:.2f}</b> remaining
&nbsp;·&nbsp; <span class="muted">generated {datetime.datetime.now():%Y-%m-%d %H:%M}</span></p>
<table><tr><th>Date</th><th>SV images</th><th>SV $</th><th>Gemini calls</th>
<th>Gemini $</th><th>Day total</th><th>Cumulative</th></tr>{trs}</table>
<p class="muted">Street View Static API @ $0.007/image; Gemini @ token list price.
Mapillary &amp; Street View metadata calls are free. GCP VM compute is NOT
included — see <a href="https://console.cloud.google.com/billing">the Billing console</a>.</p>
</body></html>"""
    html_out.write_text(html, encoding="utf-8")

    print(f"total API spend: ${total:.2f}  (of ${BUDGET:.0f} budget)")
    for d, svn, svc, gmn, gmc, tot, c in rows:
        print(f"  {d}: SV {svn} imgs ${svc:.2f} + Gemini {gmn} calls "
              f"${gmc:.2f} = ${tot:.2f}  (cum ${c:.2f})")
    print(f"log -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
