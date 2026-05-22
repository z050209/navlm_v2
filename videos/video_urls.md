# NavLM — source videos (8 YouTube Zurich walking tours)

All training data is self-supervised from these 8 unlabelled 4K walking-tour
videos of Zurich. ~49 GB total.

| # | Dataset name | YouTube title | URL |
|---|--------------|---------------|-----|
| 1 | `zurich_main` (Original) | ZURICH, Switzerland — 4K 60fps | https://www.youtube.com/watch?v=h7saB68KE5M |
| 2 | `bahnhofstrasse` | Switzerland Zurich Bahnhofstrasse Walking tour / City Center 4K 60fps HDR | https://www.youtube.com/watch?v=g21yfR4yNd8 |
| 3 | `most_famous` | Walking Tour of Switzerland's Most Famous City / Zurich 4K City Walk | https://www.youtube.com/watch?v=F8KpE5iEvW0 |
| 4 | `saturday_morning` ⭐ HOLD-OUT | Zurich looks STUNNING on Saturday Morning / Switzerland Walk [4K] | https://www.youtube.com/watch?v=8zcXNiWRgtA |
| 5 | `looks_perfect` | Zurich, Switzerland / A City That Looks Too Perfect to Be Real / 4K Walking Tour | https://www.youtube.com/watch?v=3BnA_kP2HHY |
| 6 | `old_town_limmat` | Zurich, Switzerland Old Town & Limmat River / Walking Tour / 4K Ambient | https://www.youtube.com/watch?v=JUuggKe733s |
| 7 | `most_elegant` | ZURICH, Switzerland The Most Elegant City in Europe? / 4K Walking Tour | https://www.youtube.com/watch?v=5175ziTF3Gc |
| 8 | `hidden_streets` | Zurich in Summer / Hidden Streets, River Views & Swiss Perfection [4K HDR 60FPS] | https://www.youtube.com/watch?v=QU1HxFTuqPY |

## Download

```bash
# yt-dlp, best quality mp4
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
    -o "%(id)s.mp4" <URL>
```

`saturday_morning` (#4) is the evaluation hold-out — never used in training.
