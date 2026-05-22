"""Fetch walking-tour videos for a given city.

Priority (first source that yields >=1 video wins):
  1. --url-list: user-supplied URLs
  2. --local-dir: directory already containing .mp4 files
  3. yt-dlp search on Bilibili using config queries (may be blocked)
  4. yt-dlp on the config's youtube_seed URL (blocked from this host)

Outputs into data/cities/{city}/videos/*.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/pub/evaluation_group/ning/test/navlm")
sys.path.insert(0, str(ROOT / "toolbox"))
from config import CITIES  # noqa


def fetch_from_urls(urls, dest):
    ok = []
    for url in urls:
        cmd = ["yt-dlp", "-f", "best[ext=mp4]/best", "-o", str(dest / "%(id)s.%(ext)s"), url]
        try:
            subprocess.run(cmd, check=True, timeout=1800)
            ok.append(url)
        except subprocess.CalledProcessError as e:
            print(f"[fetch] yt-dlp failed {url}: {e}")
        except subprocess.TimeoutExpired:
            print(f"[fetch] yt-dlp timeout {url}")
    return ok


def fetch_from_local(local_dir, dest):
    src = Path(local_dir)
    if not src.exists():
        return []
    videos = list(src.glob("*.mp4")) + list(src.glob("*.mov")) + list(src.glob("*.webm"))
    out = []
    for v in videos:
        target = dest / v.name
        if not target.exists():
            os.symlink(v.resolve(), target)
        out.append(str(target))
    return out


def fetch_from_bili(queries, dest, max_per_query=1):
    ok = []
    for q in queries:
        search_url = f"bilisearch{max_per_query}:{q}"
        cmd = ["yt-dlp", "-f", "best[ext=mp4]/best", "-o", str(dest / "%(id)s.%(ext)s"),
               search_url]
        try:
            subprocess.run(cmd, check=True, timeout=600)
            ok.append(q)
        except Exception as e:
            print(f"[fetch] bili search failed for {q}: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, help="city id in config.CITIES")
    ap.add_argument("--url-list", type=str, help="text file of one URL per line")
    ap.add_argument("--local-dir", type=str, help="pre-downloaded mp4 directory")
    ap.add_argument("--skip-network", action="store_true", help="don't try network sources")
    args = ap.parse_args()

    if args.city not in CITIES:
        raise SystemExit(f"Unknown city {args.city}. Known: {list(CITIES)}")

    dest = ROOT / "data" / "cities" / args.city / "videos"
    dest.mkdir(parents=True, exist_ok=True)

    got = []
    if args.url_list:
        urls = [ln.strip() for ln in open(args.url_list) if ln.strip() and not ln.startswith("#")]
        got.extend(fetch_from_urls(urls, dest))

    if not got and args.local_dir:
        got.extend(fetch_from_local(args.local_dir, dest))

    if not got and not args.skip_network:
        city_cfg = CITIES[args.city]
        got.extend(fetch_from_bili(city_cfg["search_queries"], dest))
        if not got and city_cfg.get("youtube_seed"):
            got.extend(fetch_from_urls([city_cfg["youtube_seed"]], dest))

    got_files = list(dest.glob("*.mp4")) + list(dest.glob("*.mov")) + list(dest.glob("*.webm"))
    print(f"[fetch] {len(got_files)} videos available in {dest}")
    for f in got_files:
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    if not got_files:
        print("[fetch] ⚠ no videos — either provide --local-dir / --url-list "
              "or rely on --static-image-dir mode in the orchestrator.")
        sys.exit(2)


if __name__ == "__main__":
    main()
