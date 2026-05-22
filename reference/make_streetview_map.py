#!/usr/bin/env python3
"""Render a Leaflet map comparing reference-image coverage in Zurich.

Self-contained — no dependencies. Toggleable layers (top-right control):
  - Mapillary 5k      : the current Phase-A reference index
  - SV scanned        : all 1,915 Street View panoramas found (free scan)
  - SV downloaded     : the 178 panoramas downloaded ($5 test batch)
  - POI shots         : the 25-POI Street View images

Run from navlm_ss/:
    python make_streetview_map.py
"""

import csv
import json
from pathlib import Path

MLY_META = Path("data/cities/mapillary/zurich/meta.jsonl")
PANOS = Path("data/cities/streetview/zurich/panos.jsonl")
META = Path("data/cities/streetview/zurich/meta.jsonl")
POI_MANIFEST = Path("../preview/streetview_poi/manifest.csv")
OUT = Path("../preview/streetview_map.html")


def bbox(pts):
    la = [p["lat"] for p in pts]
    lo = [p["lon"] for p in pts]
    return min(la), min(lo), max(la), max(lo)


def main():
    mly = [{"lat": d["lat"], "lon": d["lon"], "id": d["id"]}
           for d in (json.loads(l) for l in MLY_META.open(encoding="utf-8"))]

    scanned = [json.loads(l) for l in PANOS.open(encoding="utf-8")]
    dl_count = {}
    for l in META.open(encoding="utf-8"):
        d = json.loads(l)
        dl_count[d["pano_id"]] = dl_count.get(d["pano_id"], 0) + 1
    downloaded, scanned_only = [], []
    for p in scanned:
        rec = {"lat": p["lat"], "lon": p["lon"], "id": p["pano_id"],
               "date": p.get("date", ""), "imgs": dl_count.get(p["pano_id"], 0)}
        (downloaded if rec["imgs"] else scanned_only).append(rec)

    pois = []
    if POI_MANIFEST.exists():
        for r in csv.DictReader(POI_MANIFEST.open(encoding="utf-8")):
            pois.append({"lat": float(r["pano_lat"]), "lon": float(r["pano_lon"]),
                         "poi": r["poi"], "sample": r["sample"]})

    clat = sum(p["lat"] for p in scanned) / len(scanned)
    clon = sum(p["lon"] for p in scanned) / len(scanned)
    mb = bbox(mly)
    sb = bbox(scanned)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Zurich reference-image coverage</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}
.legend{{background:#fff;padding:8px 12px;font:12px sans-serif;line-height:18px;
box-shadow:0 0 8px rgba(0,0,0,.3);border-radius:4px}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}}</style>
</head><body><div id="map"></div><script>
var map=L.map('map').setView([{clat},{clon}],15);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
 {{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(map);

var mly={json.dumps(mly)};
var scannedOnly={json.dumps(scanned_only)};
var downloaded={json.dumps(downloaded)};
var pois={json.dumps(pois)};

function layer(arr,style,popup){{
 var g=L.layerGroup();
 arr.forEach(function(p){{
  L.circleMarker([p.lat,p.lon],style).bindPopup(popup(p)).addTo(g);}});
 return g;}}

var lMly=layer(mly,{{radius:3,color:'#176',fillColor:'#3b9',fillOpacity:.5,weight:1}},
  function(p){{return 'Mapillary<br>'+p.id;}});
var lScan=layer(scannedOnly,{{radius:3,color:'#999',fillColor:'#bbb',fillOpacity:.5,weight:1}},
  function(p){{return 'SV scanned-only<br>'+p.id+'<br>'+p.date;}});
var lDl=layer(downloaded,{{radius:6,color:'#b00',fillColor:'#f33',fillOpacity:.9,weight:1}},
  function(p){{return '<b>SV DOWNLOADED</b> ('+p.imgs+' imgs)<br>'+p.id+'<br>'+p.date;}});
var lPoi=layer(pois,{{radius:6,color:'#024',fillColor:'#39f',fillOpacity:.9,weight:1}},
  function(p){{return '<b>POI: '+p.poi+'</b><br>'+p.sample;}});

lMly.addTo(map); lScan.addTo(map); lDl.addTo(map); lPoi.addTo(map);
L.control.layers(null,{{
 'Mapillary 5k ({len(mly)})':lMly,
 'SV scanned-only ({len(scanned_only)})':lScan,
 'SV downloaded ({len(downloaded)})':lDl,
 'POI shots ({len(pois)})':lPoi}},{{collapsed:false}}).addTo(map);

// bbox rectangles
L.rectangle([[{mb[0]},{mb[1]}],[{mb[2]},{mb[3]}]],
 {{color:'#3b9',weight:2,fill:false,dashArray:'6'}}).addTo(map);
L.rectangle([[{sb[0]},{sb[1]}],[{sb[2]},{sb[3]}]],
 {{color:'#999',weight:2,fill:false,dashArray:'6'}}).addTo(map);

var lg=L.control({{position:'bottomleft'}});
lg.onAdd=function(){{var d=L.DomUtil.create('div','legend');
 d.innerHTML='<b>Reference-image coverage</b><br>'+
 '<span class="dot" style="background:#3b9"></span>Mapillary 5k index<br>'+
 '<span class="dot" style="background:#bbb"></span>SV scanned-only<br>'+
 '<span class="dot" style="background:#f33"></span>SV downloaded ($5 test)<br>'+
 '<span class="dot" style="background:#39f"></span>POI Street View shots<br>'+
 'dashed boxes = each set\\'s bounding range';
 return d;}};
lg.addTo(map);
</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Mapillary 5k     : {len(mly)} images")
    print(f"  bbox lat {mb[0]:.4f}-{mb[2]:.4f}  lon {mb[1]:.4f}-{mb[3]:.4f}")
    print(f"StreetView scan  : {len(scanned)} panos ({len(downloaded)} downloaded)")
    print(f"  bbox lat {sb[0]:.4f}-{sb[2]:.4f}  lon {sb[1]:.4f}-{sb[3]:.4f}")
    print(f"POI shots        : {len(pois)}")
    print(f"map -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
