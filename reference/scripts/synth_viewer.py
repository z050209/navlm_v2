"""Web portal for spot-checking synth_unified.jsonl.

A small Flask app on port 9000 that lets you scroll through the synthesised
training samples — image + system prompt + user msg + CoT + answer all on
one page.

Features:
  - Resized, cached images (max 1024px wide) for snappy loading
  - EN ⇄ 中文 UI toggle (?lang=zh)
  - On-demand Chinese translation of system/user/thinking/answer via Gemma

Usage
-----
    python scripts/synth_viewer.py
    open http://<host>:9000

Hot-reload: jsonl is re-read on every request, so you can leave this running
while synth is generating new rows.
"""

import argparse
import io
import json
import re
import threading
from pathlib import Path

import requests
from flask import (Flask, abort, jsonify, render_template_string,
                    request, send_file)
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__)


# ─────────────── data loading ───────────────

def load_rows(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return rows


def get_thinking(text):
    m = re.search(r"<thinking>\s*(.*?)\s*</thinking>", text or "", re.S)
    return m.group(1) if m else None


def get_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", re.S)
    return m.group(1) if m else None


# ─────────────── image cache ───────────────

_IMG_CACHE = {}             # path -> resized bytes
_IMG_LOCK = threading.Lock()


def get_resized_image_bytes(path: Path, max_width: int = 1024):
    """Return JPEG bytes of the image resized to max_width (cached)."""
    key = str(path.resolve())
    with _IMG_LOCK:
        if key in _IMG_CACHE:
            return _IMG_CACHE[key]
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        new_h = int(img.height * max_width / img.width)
        img = img.resize((max_width, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    data = buf.getvalue()
    with _IMG_LOCK:
        if len(_IMG_CACHE) > 500:
            _IMG_CACHE.pop(next(iter(_IMG_CACHE)))  # FIFO drop
        _IMG_CACHE[key] = data
    return data


# ─────────────── translation cache ───────────────

_TR_CACHE = {}              # (idx, lang, field) -> translated text
_TR_LOCK = threading.Lock()


def translate_text(text: str, lang: str, vllm_url: str, model: str) -> str:
    """Call Gemma to translate `text` to `lang` ('zh' or 'en'). Stub for empty."""
    if not text or not text.strip():
        return text
    prompt = (
        "Translate the following passage to "
        + ("Simplified Chinese" if lang == "zh" else "English")
        + ". Preserve any tags such as <thinking>, <answer>, route bullets, "
        "and do NOT translate place/POI names — keep them in original "
        "form (e.g. Hauptbahnhof, Bahnhofstrasse). Output only the "
        "translated text, nothing else.\n\n"
        + text
    )
    try:
        r = requests.post(
            f"{vllm_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1200,
                "temperature": 0.0,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[translation failed: {type(e).__name__}: {e}]\n\n{text}"


# ─────────────── i18n strings ───────────────

I18N = {
    "en": {
        "title": "synth_unified viewer",
        "samples": "samples",
        "frames": "frames",
        "destinations": "destinations",
        "filter_btn": "filter",
        "all_sources": "all sources",
        "all_tiers": "all tiers",
        "tier_n": "tier {}",
        "back": "← list",
        "prev": "← prev",
        "next": "next →",
        "system_prompt": "System prompt",
        "user_msg": "User message (with tool outputs)",
        "thinking": "Assistant — thinking",
        "answer": "Assistant — answer",
        "filter_placeholder": "filter by frame/dest/text",
        "translate_btn": "Translate to 中文",
        "show_original": "Show English",
        "frame_id": "frame_id",
        "destination": "destination",
        "user_gps": "user GPS",
        "heading": "heading",
        "gps_source": "gps source",
        "distance": "distance",
        "first_action": "first action",
        "route_steps": "route steps",
        "question_template": "question",
        "verifier": "verifier",
    },
    "zh": {
        "title": "训练样本查看器",
        "samples": "条样本",
        "frames": "帧",
        "destinations": "个目的地",
        "filter_btn": "过滤",
        "all_sources": "全部来源",
        "all_tiers": "全部级别",
        "tier_n": "级 {}",
        "back": "← 列表",
        "prev": "← 上一条",
        "next": "下一条 →",
        "system_prompt": "系统提示",
        "user_msg": "用户消息（含工具输出）",
        "thinking": "Assistant — 思考过程",
        "answer": "Assistant — 答案",
        "filter_placeholder": "按 frame / 目的地 / 文本搜索",
        "translate_btn": "翻译到中文",
        "show_original": "显示英文原文",
        "frame_id": "帧 ID",
        "destination": "目的地",
        "user_gps": "用户 GPS",
        "heading": "朝向",
        "gps_source": "GPS 来源",
        "distance": "距离",
        "first_action": "第一步动作",
        "route_steps": "路径步数",
        "question_template": "提问模板",
        "verifier": "verifier",
    },
}


def t(lang, key):
    return I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))


# Per-row value translations (action verbs, verifier verdicts, sources)
ROW_VALUE_ZH = {
    # action verbs from way_planner
    "continue ahead": "继续向前",
    "continue": "继续",
    "turn left": "向左转",
    "turn right": "向右转",
    "turn around": "掉头回走",
    "go back": "原路返回",
    "go back the way you came": "原路返回",
    "arrive": "到达",
    # verifier verdicts
    "yes": "是",
    "no": "否",
    "unsure": "不确定",
    "skip": "跳过",
    # gps sources
    "ocr": "OCR 锚点",
    "visual_consensus": "视觉共识",
    "mapillary_verified": "Mapillary 验证",
}


def localize_row_value(v, lang):
    if lang != "zh":
        return v
    return ROW_VALUE_ZH.get(str(v).strip().lower(), v)


# ─────────────── templates ───────────────

INDEX_TMPL = """
<!doctype html>
<html><head><title>{{T.title}}</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, ui-sans-serif, sans-serif; margin: 0;
         background: #0d1117; color: #c9d1d9; }
  header { padding: 12px 20px; background: #161b22; border-bottom: 1px solid #30363d;
           display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 18px; }
  header .stat { font-size: 14px; color: #8b949e; }
  header .filter { margin-left: auto; }
  header input, header select, header button { background: #0d1117; border: 1px solid #30363d;
                 color: #c9d1d9; padding: 4px 8px; border-radius: 4px; }
  .lang-toggle a { color: #58a6ff; padding: 2px 8px; border: 1px solid transparent;
                    border-radius: 4px; text-decoration: none; }
  .lang-toggle a.active { border-color: #58a6ff; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 8px 14px; text-align: left; border-bottom: 1px solid #21262d; }
  th { background: #161b22; font-size: 13px; color: #8b949e; }
  tr:hover { background: #161b22; }
  td a { color: #58a6ff; text-decoration: none; }
  .tier-1 { color: #ff9800; font-weight: 600; }
  .tier-2 { color: #58a6ff; }
  .tier-3 { color: #8b949e; }
  .src-ocr { background: #1f6feb22; padding: 1px 6px; border-radius: 3px; }
  .src-vc  { background: #fb950022; padding: 1px 6px; border-radius: 3px; }
  .ver-yes { color: #56d364; }
  .ver-skip { color: #8b949e; }
  .ver-unsure { color: #d29922; }
  .pagination { margin: 16px 20px; }
  .pagination a { color: #58a6ff; padding: 4px 10px; }
</style></head>
<body>
<header>
  <h1>{{T.title}}</h1>
  <span class="stat">{{total}} {{T.samples}} · {{n_frames}} {{T.frames}} · {{n_dest}} {{T.destinations}}</span>
  <a href="/map_chooser" target="_blank" style="color:#58a6ff;text-decoration:none;">🗺 paths map</a>
  <span class="lang-toggle">
    <a href="?{{qs_with('lang','en')}}" class="{% if lang=='en' %}active{% endif %}">EN</a> |
    <a href="?{{qs_with('lang','zh')}}" class="{% if lang=='zh' %}active{% endif %}">中文</a>
  </span>
  <form class="filter" method="get">
    <input type="hidden" name="lang" value="{{lang}}" />
    <input name="q" placeholder="{{T.filter_placeholder}}" value="{{q or ''}}" />
    <select name="src">
      <option value="">{{T.all_sources}}</option>
      <option value="ocr"               {% if src=='ocr' %}selected{% endif %}>ocr</option>
      <option value="visual_consensus"  {% if src=='visual_consensus' %}selected{% endif %}>visual_consensus</option>
    </select>
    <select name="tier">
      <option value="">{{T.all_tiers}}</option>
      <option value="1" {% if tier=='1' %}selected{% endif %}>{{T.tier_n.format(1)}}</option>
      <option value="2" {% if tier=='2' %}selected{% endif %}>{{T.tier_n.format(2)}}</option>
      <option value="3" {% if tier=='3' %}selected{% endif %}>{{T.tier_n.format(3)}}</option>
    </select>
    <button>{{T.filter_btn}}</button>
  </form>
</header>

<table>
  <tr>
    <th>#</th><th>{{T.frame_id}}</th><th>{{T.gps_source}}</th>
    <th>{{T.destination}}</th><th>tier</th>
    <th>{{T.first_action}}</th><th>{{T.verifier}}</th>
    <th>{{T.distance}}</th><th>min</th>
  </tr>
  {% for r in rows %}
    {% set m = r['_meta'] %}
    {% set ver_raw = (m.visual_verifier or {}).get('verifier','skip') %}
    <tr>
      <td><a href="/sample/{{r._index}}?lang={{lang}}">{{loop.index + offset}}</a></td>
      <td><a href="/sample/{{r._index}}?lang={{lang}}">{{m.start_frame}}</a></td>
      <td><span class="src-{{ 'ocr' if m.gps_source=='ocr' else 'vc' }}">{{loc(m.gps_source)}}</span></td>
      <td>{{m.destination}}</td>
      <td class="tier-{{m.destination_tier}}">T{{m.destination_tier}}</td>
      <td>{{loc(m.first_action)}}</td>
      <td class="ver-{{ver_raw}}">{{loc(ver_raw)}}</td>
      <td>{{m.distance_m}}m</td>
      <td>{{m.estimated_minutes}}</td>
    </tr>
  {% endfor %}
</table>

<div class="pagination">
  {% if offset > 0 %}<a href="?{{qs_with('offset',offset-page_size)}}">{{T.prev}}</a>{% endif %}
  {% if offset + page_size < total %}<a href="?{{qs_with('offset',offset+page_size)}}">{{T.next}}</a>{% endif %}
</div>
</body></html>
"""

DETAIL_TMPL = """
<!doctype html>
<html><head><title>{{m.start_frame}} → {{m.destination}}</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, ui-sans-serif, sans-serif; margin: 0;
         background: #0d1117; color: #c9d1d9; }
  header { padding: 12px 20px; background: #161b22; border-bottom: 1px solid #30363d;
           display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
  header h2 { margin: 0; font-size: 16px; }
  header a { color: #58a6ff; text-decoration: none; }
  .lang-toggle { margin-left: auto; }
  .lang-toggle a { padding: 2px 8px; border: 1px solid transparent;
                    border-radius: 4px; text-decoration: none; }
  .lang-toggle a.active { border-color: #58a6ff; }
  .container { display: grid; grid-template-columns: 50% 50%; gap: 0;
               height: calc(100vh - 56px); }
  .left, .right { overflow-y: auto; padding: 16px 24px; }
  .left { background: #0d1117; border-right: 1px solid #30363d; }
  .img-wrap { background: #161b22; border-radius: 6px; padding: 8px; margin-bottom: 16px; }
  .img-wrap img { width: 100%; border-radius: 4px; }
  .meta-grid { display: grid; grid-template-columns: 130px 1fr; gap: 6px 12px;
               font-size: 13px; color: #8b949e; }
  .meta-grid b { color: #c9d1d9; }
  pre { background: #161b22; border: 1px solid #21262d; border-radius: 6px;
        padding: 12px 14px; white-space: pre-wrap; word-break: break-word;
        font-size: 13px; line-height: 1.5; }
  h3 { color: #58a6ff; margin: 18px 0 6px; font-size: 14px;
       text-transform: uppercase; letter-spacing: 0.05em; display: flex; gap: 12px; align-items: center; }
  h3 button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
               border-radius: 4px; padding: 2px 8px; font-size: 11px; cursor: pointer; }
  h3 button:hover { background: #30363d; }
  .answer { background: #173a1f; border-color: #56d36444; }
  .thinking { background: #1c2533; border-color: #1f6feb44; }
  .nav { margin-top: 16px; display: flex; gap: 12px; }
  .nav a { color: #58a6ff; padding: 6px 12px; background: #161b22;
           border-radius: 4px; text-decoration: none; }
  .loader { display: inline-block; opacity: 0.6; font-size: 11px; }
</style></head>
<body>
<header>
  <a href="/?{{back_qs}}">{{T.back}}</a>
  <h2>{{m.start_frame}} → {{m.destination}}</h2>
  <span style="color:#8b949e;font-size:13px;">
    {{m.gps_source}} · T{{m.destination_tier}} · {{m.first_action}} · {{m.distance_m}}m · {{m.estimated_minutes}}min
  </span>
  <span class="lang-toggle">
    <a href="?lang=en" class="{% if lang=='en' %}active{% endif %}">EN</a> |
    <a href="?lang=zh" class="{% if lang=='zh' %}active{% endif %}">中文</a>
  </span>
</header>

<div class="container">
  <div class="left">
    <div class="img-wrap">
      <img id="frame-img" src="/image/{{idx}}" alt="{{m.start_frame}}" loading="lazy" />
      <div style="margin-top:8px;display:flex;gap:8px;align-items:center;">
        <button id="toggle-arrow" onclick="toggle_arrow()" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px;">↗ {{ '显示箭头' if lang=='zh' else 'show arrow' }}</button>
        <button id="toggle-bbox"  onclick="toggle_bbox()" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px;">▢ {{ 'POI 框选' if lang=='zh' else 'show bbox' }}</button>
        <span id="bbox-status" style="font-size:11px;color:#8b949e;"></span>
      </div>
    </div>

    <div class="meta-grid">
      <b>{{T.frame_id}}</b><span>{{m.start_frame}}</span>
      <b>{{T.destination}}</b><span>{{m.destination}} (T{{m.destination_tier}})</span>
      <b>{{T.user_gps}}</b><span>{{m.user_gps[0]|round(5)}}, {{m.user_gps[1]|round(5)}}</span>
      <b>{{T.heading}}</b><span>{{m.user_heading}}° ({{m.heading_confidence}})</span>
      <b>{{T.gps_source}}</b><span>{{m.gps_source}}</span>
      <b>{{T.distance}}</b><span>{{m.distance_m}}m, ~{{m.estimated_minutes}}min</span>
      <b>{{T.first_action}}</b><span>{{m.first_action}}</span>
      <b>{{T.route_steps}}</b><span>{{m.n_route_steps}}</span>
      <b>{{T.question_template}}</b><span>{{m.question_template}}</span>
      {% if m.visual_verifier %}
      <b>{{T.verifier}}</b><span>
        {{m.visual_verifier.verifier}} · {{m.visual_verifier.poi_checked}}<br>
        <small>{{m.visual_verifier.verifier_raw}}</small>
      </span>
      {% endif %}
    </div>

    <div class="nav">
      {% if prev_idx is not none %}<a href="/sample/{{prev_idx}}?{{back_qs}}">{{T.prev}}</a>{% endif %}
      {% if next_idx is not none %}<a href="/sample/{{next_idx}}?{{back_qs}}">{{T.next}}</a>{% endif %}
    </div>
  </div>

  <div class="right">
    <h3>{{T.system_prompt}} <button onclick="tr('system')">{{T.translate_btn if lang=='en' else T.show_original}}</button>
        <span id="ld-system" class="loader" style="display:none;">…</span></h3>
    <pre id="text-system">{{system_msg}}</pre>

    <h3>{{T.user_msg}} <button onclick="tr('user')">{{T.translate_btn if lang=='en' else T.show_original}}</button>
        <span id="ld-user" class="loader" style="display:none;">…</span></h3>
    <pre id="text-user">{{user_msg}}</pre>

    <h3>{{T.thinking}} <button onclick="tr('thinking')">{{T.translate_btn if lang=='en' else T.show_original}}</button>
        <span id="ld-thinking" class="loader" style="display:none;">…</span></h3>
    <pre class="thinking" id="text-thinking">{{thinking or '(missing)'}}</pre>

    <h3>{{T.answer}}
        <button onclick="tr('answer')">{{T.translate_btn if lang=='en' else T.show_original}}</button>
        <button onclick="play_tts()">🔊 {{ '播放语音' if lang=='zh' else 'Play voice' }}</button>
        <span id="ld-answer" class="loader" style="display:none;">…</span>
    </h3>
    <audio id="tts-audio" controls preload="none"
           style="width:100%;margin-bottom:8px;display:none"></audio>
    <pre class="answer" id="text-answer">{{answer or '(missing)'}}</pre>
  </div>
</div>

<script>
const IDX = {{idx}};
const PAGE_LANG = "{{lang}}";
const originals = {
  system:    {{system_msg|tojson}},
  user:      {{user_msg|tojson}},
  thinking:  {{(thinking or '')|tojson}},
  answer:    {{(answer or '')|tojson}},
};
const translated = {};

function showLoading(field, on) {
  document.getElementById('ld-' + field).style.display = on ? 'inline-block' : 'none';
}

function fetchTranslation(field) {
  if (translated[field]) return Promise.resolve(translated[field]);
  showLoading(field, true);
  return fetch(`/translate/${IDX}/${field}?lang=zh`)
    .then(r => r.json())
    .then(d => { translated[field] = d.text; return d.text; })
    .finally(() => showLoading(field, false));
}

// Manual toggle button per section
function tr(field) {
  const elText = document.getElementById('text-' + field);
  if (elText.dataset.showing === 'zh') {
    elText.textContent = originals[field];
    elText.dataset.showing = 'en';
    return;
  }
  fetchTranslation(field).then(text => {
    elText.textContent = text;
    elText.dataset.showing = 'zh';
  });
}

// Server-side translation handles auto-translation when lang=zh.
// JS is used only for the manual toggle button on each section.
for (const field of ['system','user','thinking','answer']) {
  const el = document.getElementById('text-' + field);
  if (el) el.dataset.showing = PAGE_LANG;
}

// gTTS playback
function play_tts() {
  const audio = document.getElementById('tts-audio');
  audio.style.display = 'block';
  audio.src = `/tts/${IDX}?lang=${PAGE_LANG}`;
  audio.play().catch(e => alert('audio error: ' + e));
}

// Arrow toggle on frame image
let _arrow_on = false;
function toggle_arrow() {
  _arrow_on = !_arrow_on;
  const img = document.getElementById('frame-img');
  img.src = `/image/${IDX}` + (_arrow_on ? `?arrow=1&t=${Date.now()}` : `?t=${Date.now()}`);
  document.getElementById('toggle-arrow').textContent = _arrow_on
    ? '↗ ' + (PAGE_LANG === 'zh' ? '隐藏箭头' : 'hide arrow')
    : '↗ ' + (PAGE_LANG === 'zh' ? '显示箭头' : 'show arrow');
}

// BBox toggle: ask Gemma to locate the visible USER_LOCATION POI
let _bbox_on = false;
function toggle_bbox() {
  _bbox_on = !_bbox_on;
  const img = document.getElementById('frame-img');
  const status = document.getElementById('bbox-status');
  if (!_bbox_on) {
    img.src = `/image/${IDX}?t=${Date.now()}`;
    status.textContent = '';
    document.getElementById('toggle-bbox').textContent = '▢ ' +
      (PAGE_LANG === 'zh' ? 'POI 框选' : 'show bbox');
    return;
  }
  status.textContent = PAGE_LANG === 'zh' ? '正在定位 POI...' : 'finding POI...';
  img.src = `/image/${IDX}?bbox=1&t=${Date.now()}`;
  img.onload = () => {
    status.textContent = PAGE_LANG === 'zh' ? '✓ 定位完成' : '✓ done';
    document.getElementById('toggle-bbox').textContent = '▢ ' +
      (PAGE_LANG === 'zh' ? '隐藏 bbox' : 'hide bbox');
  };
}
</script>
</body></html>
"""


# ─────────────── routes ───────────────

def qs_with_factory(args):
    """Return a function that produces a query-string with one key overridden."""
    def f(key, value):
        from urllib.parse import urlencode
        kept = {k: v for k, v in args.items() if v}
        kept[key] = value
        return urlencode(kept)
    return f


@app.route("/frame_image/<path:rel>")
def frame_image(rel):
    """Serve a resized frame from data/cities/zurich/frames/...
    `rel` is a path relative to the frames dir, e.g. "extra_X/frame_00123.jpg"."""
    p = ROOT / "data/cities/zurich/frames" / rel
    if not p.is_file():
        abort(404)
    data = get_resized_image_bytes(p, max_width=480)
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


@app.route("/map")
def map_view():
    """Serve the path-visualization HTML. Optional ?density=300 to re-render
    with more / fewer polyline points per video."""
    density = request.args.get("density")
    if density:
        try:
            n = max(20, min(2000, int(density)))
            import subprocess as _sp
            _sp.run(["python", str(ROOT / "scripts/visualize_paths.py"),
                     "--density", str(n)], check=True, cwd=str(ROOT))
        except Exception as e:
            return f"render failed: {e}", 500
    p = ROOT / "data/cities/zurich/_paths_map.html"
    if not p.exists():
        return ("Map not built yet. Run: "
                "python scripts/visualize_paths.py", 404)
    return send_file(p, mimetype="text/html")


@app.route("/map_chooser")
def map_chooser():
    """Page with a density slider that re-renders the map on change."""
    return render_template_string("""
<!doctype html>
<html><head><title>paths map</title>
<style>
  body { margin:0; font-family: ui-sans-serif, sans-serif; background:#0d1117; color:#c9d1d9; }
  .top { padding:8px 16px; background:#161b22; border-bottom:1px solid #30363d;
         display:flex; gap:16px; align-items:center; }
  iframe { border:0; width:100%; height:calc(100vh - 50px); }
  input[type=range] { width: 320px; }
  .badge { color:#8b949e; font-size:13px; }
</style></head>
<body>
<div class="top">
  <b>Path map</b>
  <span class="badge">density (points per video):</span>
  <input id="den" type="range" min="40" max="600" step="20" value="150"
         oninput="document.getElementById('lbl').textContent=this.value;
                  document.getElementById('frm').src='/map?density='+this.value;" />
  <span id="lbl" class="badge">150</span>
  <span class="badge">| 40 = sparse, clean &nbsp;·&nbsp; 600 = dense, every step</span>
  <a href="/" class="badge" style="margin-left:auto;color:#58a6ff;text-decoration:none;">← back</a>
</div>
<iframe id="frm" src="/map"></iframe>
</body></html>""")


@app.route("/")
def index():
    rows = load_rows(app.config["JSONL"])
    for i, r in enumerate(rows):
        r["_index"] = i

    q = (request.args.get("q") or "").strip().lower()
    src = request.args.get("src") or ""
    tier = request.args.get("tier") or ""
    offset = int(request.args.get("offset") or 0)
    lang = _normalize_lang(request.args.get("lang"))
    page_size = 50

    filtered = rows
    if q:
        def match(r):
            m = r.get("_meta", {})
            blob = " ".join(str(v).lower() for v in [
                m.get("start_frame", ""),
                m.get("destination", ""),
                m.get("first_action", ""),
                m.get("question_template", ""),
            ])
            return q in blob
        filtered = [r for r in filtered if match(r)]
    if src:
        filtered = [r for r in filtered if r.get("_meta", {}).get("gps_source") == src]
    if tier:
        filtered = [r for r in filtered if str(r.get("_meta", {}).get("destination_tier")) == tier]

    total = len(filtered)
    page = filtered[offset: offset + page_size]
    n_frames = len({r["_meta"]["start_frame"] for r in rows if r.get("_meta")})
    n_dest = len({r["_meta"]["destination"] for r in rows if r.get("_meta")})

    return render_template_string(
        INDEX_TMPL,
        rows=page, total=total,
        offset=offset, page_size=page_size,
        n_frames=n_frames, n_dest=n_dest,
        q=q, src=src, tier=tier, lang=lang,
        T=I18N.get(lang, I18N["en"]),
        qs_with=qs_with_factory(request.args),
        loc=lambda v: localize_row_value(v, lang),
    )


def _normalize_lang(s):
    """Accept zh / zh-CN / cn / chinese / 中文 etc. Default = en."""
    if not s:
        return "en"
    s = s.strip().lower()
    zh_aliases = {"zh", "zn", "cn", "chinese", "china", "zh-cn", "zh-hans",
                   "中文", "中"}
    return "zh" if s in zh_aliases or s.startswith("zh") else "en"


def _translate_all(idx, lang, fields):
    """Translate the given fields in parallel (with content-hash cache).

    Cache key is (lang, hash(text)) — so the SYSTEM_PROMPT (identical
    across samples) is translated once and reused everywhere.
    """
    from concurrent.futures import ThreadPoolExecutor
    import hashlib

    def cache_key(text):
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        return (lang, h)

    def one(name, text):
        if not text:
            return name, ""
        ck = cache_key(text)
        with _TR_LOCK:
            if ck in _TR_CACHE:
                return name, _TR_CACHE[ck]
        out = translate_text(text, lang,
                              vllm_url=app.config["VLLM_URL"],
                              model=app.config["MODEL"])
        with _TR_LOCK:
            _TR_CACHE[ck] = out
        return name, out

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(one, n, t) for n, t in fields.items()]
        results = {f.result()[0]: f.result()[1] for f in futures}
    return results


@app.route("/sample/<int:idx>")
def sample(idx):
    rows = load_rows(app.config["JSONL"])
    if idx < 0 or idx >= len(rows):
        abort(404)
    r = rows[idx]
    msgs = r.get("messages", [])
    sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    thinking = get_thinking(asst_msg)
    answer = get_answer(asst_msg)
    lang = _normalize_lang(request.args.get("lang"))

    # Server-side translate when lang=zh — much more reliable than JS.
    if lang == "zh":
        translated = _translate_all(idx, "zh", {
            "system": sys_msg, "user": user_msg,
            "thinking": thinking or "", "answer": answer or "",
        })
        sys_msg = translated["system"]
        user_msg = translated["user"]
        thinking = translated["thinking"] or thinking
        answer = translated["answer"] or answer

    back_qs = "&".join(f"{k}={v}" for k, v in request.args.items() if v)

    return render_template_string(
        DETAIL_TMPL,
        idx=idx, m=r["_meta"],
        system_msg=sys_msg,
        user_msg=user_msg,
        thinking=thinking,
        answer=answer,
        prev_idx=idx - 1 if idx > 0 else None,
        next_idx=idx + 1 if idx < len(rows) - 1 else None,
        back_qs=back_qs,
        lang=lang,
        T=I18N.get(lang, I18N["en"]),
    )


@app.route("/image/<int:idx>")
def image(idx):
    rows = load_rows(app.config["JSONL"])
    if idx < 0 or idx >= len(rows):
        abort(404)
    p = Path(rows[idx]["image"])
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        abort(404)

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "toolbox"))
    m = rows[idx]["_meta"]

    if request.args.get("arrow") == "1":
        from draw_direction_arrow import draw_arrow
        data = draw_arrow(
            p,
            first_action=m.get("first_action"),
            user_heading=m.get("user_heading"),
            label=m.get("first_action"),
        )
    elif request.args.get("bbox") == "1":
        from draw_poi_bbox import draw_bbox
        # Ask the verifier-checked POI (the one we claimed user is at)
        poi_name = (m.get("visual_verifier") or {}).get("poi_checked") \
                   or m.get("destination") \
                   or "the destination"
        data = draw_bbox(p, poi_name,
                         vllm_url=app.config["VLLM_URL"],
                         model=app.config["MODEL"])
    else:
        data = get_resized_image_bytes(p, max_width=1024)
    return send_file(io.BytesIO(data), mimetype="image/jpeg",
                     download_name=f"{rows[idx]['_meta']['start_frame']}.jpg")


_TTS_CACHE = {}     # (idx, lang) -> mp3 bytes
_TTS_LOCK = threading.Lock()


@app.route("/tts/<int:idx>")
def tts(idx):
    """Synthesize speech for the answer of sample idx. ?lang=en|zh."""
    from flask import Response
    rows = load_rows(app.config["JSONL"])
    if idx < 0 or idx >= len(rows):
        abort(404)
    lang = _normalize_lang(request.args.get("lang"))
    cache_key = (idx, lang)
    with _TTS_LOCK:
        if cache_key in _TTS_CACHE:
            return Response(_TTS_CACHE[cache_key], mimetype="audio/mpeg")

    r = rows[idx]
    asst = next((m["content"] for m in r.get("messages", [])
                  if m["role"] == "assistant"), "")
    text = get_answer(asst) or ""
    if lang == "zh":
        # If we already cached the translated answer, reuse it
        import hashlib
        ck = ("zh", hashlib.sha1(text.encode()).hexdigest()[:16])
        with _TR_LOCK:
            if ck in _TR_CACHE:
                text = _TR_CACHE[ck]
            else:
                text = translate_text(text, "zh",
                                       vllm_url=app.config["VLLM_URL"],
                                       model=app.config["MODEL"])
                _TR_CACHE[ck] = text
    if not text.strip():
        abort(404)

    try:
        from gtts import gTTS
        import io as _io
        t = gTTS(text=text, lang="zh-CN" if lang == "zh" else "en", slow=False)
        buf = _io.BytesIO()
        t.write_to_fp(buf)
        data = buf.getvalue()
    except Exception as e:
        return f"tts error: {type(e).__name__}: {e}", 500
    with _TTS_LOCK:
        if len(_TTS_CACHE) > 200:
            _TTS_CACHE.pop(next(iter(_TTS_CACHE)))
        _TTS_CACHE[cache_key] = data
    return Response(data, mimetype="audio/mpeg")


@app.route("/translate/<int:idx>/<field>")
def translate(idx, field):
    if field not in {"system", "user", "thinking", "answer"}:
        abort(400)
    rows = load_rows(app.config["JSONL"])
    if idx < 0 or idx >= len(rows):
        abort(404)
    r = rows[idx]
    msgs = r.get("messages", [])
    if field == "system":
        text = next((m["content"] for m in msgs if m["role"] == "system"), "")
    elif field == "user":
        text = next((m["content"] for m in msgs if m["role"] == "user"), "")
    else:
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        text = (get_thinking(asst) if field == "thinking" else get_answer(asst)) or ""

    lang = request.args.get("lang") or "zh"
    cache_key = (idx, lang, field)
    with _TR_LOCK:
        if cache_key in _TR_CACHE:
            return jsonify({"text": _TR_CACHE[cache_key]})

    out = translate_text(
        text, lang,
        vllm_url=app.config["VLLM_URL"],
        model=app.config["MODEL"],
    )
    with _TR_LOCK:
        _TR_CACHE[cache_key] = out
    return jsonify({"text": out})


# ─────────────── experiment views ───────────────

EVAL_TAGS = [
    ("base_v3",  "A1 base+v3"),
    ("base_v4a", "A2 base+v4a"),
    ("base_v4b", "A3 base+v4b"),
    ("lora_v3",  "C1 LoRA+v3"),
    ("lora_v4a", "C2 LoRA+v4a"),
    ("lora_v4b", "C3 LoRA+v4b"),
    ("lora_v4c", "C4 LoRA+v4c"),
]


def _load_eval_data():
    """Load all per-row eval results, indexed by frame_id × tag."""
    by_frame = {}
    summaries = {}
    for tag, _ in EVAL_TAGS:
        path = ROOT / f"results/eval_v3_{tag}.jsonl"
        if not path.exists():
            continue
        rows = []
        with open(path) as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        for r in rows:
            by_frame.setdefault(r["frame"], {})[tag] = r
        summary_path = ROOT / f"results/eval_v3_{tag}.json"
        if summary_path.exists():
            try:
                summaries[tag] = json.load(open(summary_path))
            except Exception:
                pass
    return by_frame, summaries


@app.route("/experiment_summary")
def experiment_summary():
    """Dashboard with the 4 plots + headline table."""
    _, summaries = _load_eval_data()
    rows = []
    for tag, label in EVAL_TAGS:
        if tag not in summaries:
            continue
        s = summaries[tag]
        g = s["gate_pass_rate"]
        d = s["delta_distribution"]
        rows.append({
            "tag": tag, "label": label,
            "n": s["n_samples"],
            "g1": g.get("1_format", 0) * 100,
            "g2": g.get("2_sentence_count", 0) * 100,
            "g3": g.get("3_closed_loop", 0) * 100,
            "g4": g.get("4_checkpoint", 0) * 100,
            "g5": g.get("5_dest_correct", 0) * 100,
            "g6": g.get("6_anchor_grounded", 0) * 100,
            "pass_strict": s.get("pass_strict_30", 0) * 100,
            "median_delta": d.get("median") or 0,
        })
    plots = [
        ("plot_pass_strict.png", "Overall pass rate (5 core gates + δ<30°)"),
        ("plot_closed_loop.png", "Closed-loop math correctness"),
        ("plot_gate_pass_rates.png", "Gate-by-gate breakdown"),
        ("plot_delta_distribution.png", "δ distribution"),
    ]
    return render_template_string(SUMMARY_TPL, rows=rows, plots=plots)


@app.route("/experiment_plot/<name>")
def experiment_plot(name):
    """Serve plot PNGs."""
    p = ROOT / "results" / name
    if not p.exists() or ".." in name:
        abort(404)
    return send_file(str(p), mimetype="image/png")


@app.route("/experiment_image/<frame_id>")
def experiment_image(frame_id):
    """Serve the hold-out frame image by frame_id."""
    by_frame, _ = _load_eval_data()
    if frame_id not in by_frame:
        abort(404)
    # Get first available row to grab image path (all 6 should point at same image)
    first = next(iter(by_frame[frame_id].values()))
    # The eval rows store frame name but image path requires looking up the row from synth_v3_eval
    eval_path = ROOT / "data/cities/zurich/synth_v3_eval.jsonl"
    for ln in open(eval_path):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r["_meta"]["start_frame"] == frame_id:
            buf = get_resized_image_bytes(Path(r["image"]), 1024)
            return send_file(io.BytesIO(buf), mimetype="image/jpeg")
    abort(404)


@app.route("/experiment")
def experiment_list():
    """List view: all 255 hold-out frames with quick gate badges."""
    by_frame, _ = _load_eval_data()
    items = []
    for fid in sorted(by_frame.keys(), key=lambda s: int(re.search(r"\d+", s).group(0))):
        per_tag = by_frame[fid]
        # Compute summary: how many of 6 conditions pass closed_loop
        n_loop_pass = sum(1 for tag, _ in EVAL_TAGS if tag in per_tag
                           and per_tag[tag]["gates"].get("3_closed_loop", {}).get("ok"))
        n_total = sum(1 for tag, _ in EVAL_TAGS if tag in per_tag)
        # Get destination
        first = next(iter(per_tag.values()))
        items.append({
            "frame_id": fid,
            "destination": first.get("destination", "?"),
            "heading_gt": first.get("heading_gt", 0),
            "n_loop_pass": n_loop_pass,
            "n_total": n_total,
        })
    return render_template_string(LIST_TPL, items=items)


@app.route("/experiment/<frame_id>")
def experiment_detail(frame_id):
    """Per-frame side-by-side view: image + 6 model outputs."""
    by_frame, _ = _load_eval_data()
    if frame_id not in by_frame:
        abort(404)
    per_tag = by_frame[frame_id]
    rows = []
    for tag, label in EVAL_TAGS:
        if tag not in per_tag:
            continue
        r = per_tag[tag]
        gates = r["gates"]
        rows.append({
            "tag": tag, "label": label,
            "response": r.get("model_response", ""),
            "answer": r.get("answer") or "",
            "thinking": r.get("thinking") or "",
            "delta": r.get("delta"),
            "parsed_action": r.get("parsed_action"),
            "g1": gates.get("1_format", {"ok": False, "reason": ""}),
            "g2": gates.get("2_sentence_count", {"ok": False, "reason": ""}),
            "g3": gates.get("3_closed_loop", {"ok": False, "reason": ""}),
            "g4": gates.get("4_checkpoint", {"ok": False, "reason": ""}),
            "g5": gates.get("5_dest_correct", {"ok": False, "reason": ""}),
            "g6": gates.get("6_anchor_grounded", {"ok": False, "reason": ""}),
        })
    first = next(iter(per_tag.values()))
    meta = {
        "frame_id": frame_id,
        "destination": first.get("destination", ""),
        "heading_gt": first.get("heading_gt", 0),
        "first_seg_bearing": first.get("first_seg_bearing", 0),
        "planner_action": first.get("planner_action", ""),
    }
    # Find prev / next
    all_frames = sorted(by_frame.keys(), key=lambda s: int(re.search(r"\d+", s).group(0)))
    cur_idx = all_frames.index(frame_id) if frame_id in all_frames else 0
    prev_fid = all_frames[max(0, cur_idx - 1)]
    next_fid = all_frames[min(len(all_frames) - 1, cur_idx + 1)]
    return render_template_string(DETAIL_TPL, meta=meta, rows=rows,
                                    prev_fid=prev_fid, next_fid=next_fid,
                                    cur=cur_idx + 1, total=len(all_frames))


SUMMARY_TPL = """
<!doctype html>
<html><head><title>NavLM experiment summary</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1280px; margin: 1em auto; padding: 0 1em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
th { background: #f5f5f5; }
td:first-child, th:first-child { text-align: left; font-weight: 600; }
.high { background: #e0f7e0; font-weight: 600; }
.low { background: #ffe5e0; }
img { max-width: 100%; border: 1px solid #ddd; margin: 1em 0; }
nav a { margin-right: 1em; }
h2 { margin-top: 2em; }
</style></head>
<body>
<nav>
  <a href="/">← samples</a>
  <a href="/map">map</a>
  <a href="/experiment_summary"><b>summary</b></a>
  <a href="/experiment">per-frame</a>
</nav>
<h1>Experiment results</h1>
<p>6 conditions evaluated on 255 hold-out samples (saturday_morning video).</p>
<table>
<tr><th>EXP</th><th>n</th><th>format</th><th>sentences</th><th>closed-loop</th>
    <th>checkpoint</th><th>dest correct</th><th>anchor</th>
    <th>PASS_strict</th><th>median δ</th></tr>
{% for r in rows %}
<tr>
  <td>{{r.label}}</td>
  <td>{{r.n}}</td>
  <td>{{ "%.1f%%" % r.g1 }}</td>
  <td>{{ "%.1f%%" % r.g2 }}</td>
  <td class="{% if r.g3 > 80 %}high{% elif r.g3 < 35 %}low{% endif %}">
    {{ "%.1f%%" % r.g3 }}</td>
  <td>{{ "%.1f%%" % r.g4 }}</td>
  <td>{{ "%.1f%%" % r.g5 }}</td>
  <td>{{ "%.1f%%" % r.g6 }}</td>
  <td class="{% if r.pass_strict > 80 %}high{% elif r.pass_strict < 30 %}low{% endif %}">
    {{ "%.1f%%" % r.pass_strict }}</td>
  <td>{{ "%.1f°" % r.median_delta }}</td>
</tr>
{% endfor %}
</table>
{% for src, caption in plots %}
<h2>{{ caption }}</h2>
<img src="/experiment_plot/{{src}}" alt="{{src}}" />
{% endfor %}
</body></html>
"""


LIST_TPL = """
<!doctype html>
<html><head><title>NavLM experiment per-frame</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 1em auto; padding: 0 1em; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px 10px; }
th { background: #f5f5f5; text-align: left; }
tr:hover { background: #f9f9f9; }
.score { text-align: center; font-weight: 600; }
.good { color: #2e7d32; }
.bad { color: #c62828; }
nav a { margin-right: 1em; }
</style></head>
<body>
<nav>
  <a href="/">← samples</a>
  <a href="/map">map</a>
  <a href="/experiment_summary">summary</a>
  <a href="/experiment"><b>per-frame</b></a>
</nav>
<h1>Per-frame experiment comparison</h1>
<p>{{items|length}} hold-out frames. Click any frame to see all 6 models'
   side-by-side outputs.</p>
<table>
<tr><th>frame</th><th>destination</th><th>heading_gt</th><th>closed-loop pass / total</th></tr>
{% for it in items %}
<tr>
  <td><a href="/experiment/{{it.frame_id}}">{{it.frame_id}}</a></td>
  <td>{{it.destination}}</td>
  <td>{{ "%.0f°" % it.heading_gt }}</td>
  <td class="score">
    <span class="{% if it.n_loop_pass >= it.n_total / 2 %}good{% else %}bad{% endif %}">
      {{it.n_loop_pass}}/{{it.n_total}}
    </span>
  </td>
</tr>
{% endfor %}
</table>
</body></html>
"""


DETAIL_TPL = """
<!doctype html>
<html><head><title>{{meta.frame_id}}</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 1em auto; padding: 0 1em; }
.meta { background: #f5f5f5; padding: 0.5em 1em; border-radius: 6px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
         gap: 12px; margin-top: 1em; }
.card { border: 1px solid #ddd; border-radius: 6px; padding: 10px;
        background: #fff; }
.card.pass { border-left: 4px solid #2e7d32; }
.card.fail { border-left: 4px solid #c62828; }
.card h3 { margin: 0 0 6px 0; font-size: 14px; display: flex;
           justify-content: space-between; }
.card .badge { font-size: 11px; padding: 2px 6px; border-radius: 3px; }
.card .badge.ok { background: #e0f7e0; color: #2e7d32; }
.card .badge.no { background: #ffe5e0; color: #c62828; }
.card pre { white-space: pre-wrap; font-size: 12px; max-height: 240px;
            overflow-y: auto; background: #fafafa; padding: 6px;
            border: 1px solid #eee; }
.card .gates { font-size: 11px; margin-top: 4px; }
.card .gates span { display: inline-block; padding: 2px 5px; margin: 1px;
                     border-radius: 3px; }
.gates .ok { background: #e0f7e0; }
.gates .no { background: #ffe5e0; }
img.frame { max-width: 100%; border: 1px solid #ddd; }
nav a { margin-right: 1em; }
</style></head>
<body>
<nav>
  <a href="/">← samples</a>
  <a href="/map">map</a>
  <a href="/experiment_summary">summary</a>
  <a href="/experiment">per-frame list</a>
  &nbsp; &nbsp;
  <a href="/experiment/{{prev_fid}}">‹ prev</a>
  ({{cur}}/{{total}})
  <a href="/experiment/{{next_fid}}">next ›</a>
</nav>
<h1>{{meta.frame_id}} → {{meta.destination}}</h1>
<div class="meta">
  <b>heading_gt</b>: {{ "%.1f°" % meta.heading_gt }} &nbsp;|&nbsp;
  <b>first_seg_bearing</b>: {{ "%.1f°" % meta.first_seg_bearing }} &nbsp;|&nbsp;
  <b>planner_action</b>: {{meta.planner_action}}
</div>
<img class="frame" src="/experiment_image/{{meta.frame_id}}" />
<div class="cards">
{% for r in rows %}
  {% set passed = r.g1.ok and r.g2.ok and r.g3.ok and r.g4.ok and r.g6.ok %}
  <div class="card {% if passed %}pass{% else %}fail{% endif %}">
    <h3>
      {{r.label}}
      <span class="badge {% if passed %}ok{% else %}no{% endif %}">
        {% if passed %}PASS{% else %}FAIL{% endif %}
      </span>
    </h3>
    <div><b>action:</b> {{r.parsed_action or "—"}}
         &nbsp;|&nbsp; <b>δ:</b>
         {% if r.delta is not none %}{{ "%.1f°" % r.delta }}{% else %}—{% endif %}</div>
    <pre>{{r.answer}}</pre>
    <div class="gates">
      <span class="{% if r.g1.ok %}ok{% else %}no{% endif %}">format</span>
      <span class="{% if r.g2.ok %}ok{% else %}no{% endif %}">sent</span>
      <span class="{% if r.g3.ok %}ok{% else %}no{% endif %}">loop</span>
      <span class="{% if r.g4.ok %}ok{% else %}no{% endif %}">chkpt</span>
      <span class="{% if r.g5.ok %}ok{% else %}no{% endif %}">dest</span>
      <span class="{% if r.g6.ok %}ok{% else %}no{% endif %}">anchor</span>
    </div>
  </div>
{% endfor %}
</div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",
                    default=str(ROOT / "data/cities/zurich/synth_unified.jsonl"))
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    args = ap.parse_args()

    app.config["JSONL"] = args.jsonl
    app.config["VLLM_URL"] = args.vllm_url
    app.config["MODEL"] = args.model
    print(f"[viewer] reading {args.jsonl}")
    print(f"[viewer] translation backend: {args.model} @ {args.vllm_url}")
    print(f"[viewer] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
