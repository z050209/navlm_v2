"""Unified Zurich GPS-recovery pipeline.

Standardizes the 8-video flow:

  raw frames  →  visual match  →  refine  →  ocr  →  ocr_match  →
  merge_gps  →  hmm  →  vlm_poi_scan  →  vlm_verify  →  trusted_starts

Each step reads from `data/cities/zurich/pipeline/<video>/step_NN_*.jsonl`
and writes its successor file to the same directory. The `Original`
video is referred to as `zurich_main` here for naming consistency.

Run all videos / all steps:

    python -m pipeline.run_all --from-step 7

Run one step on one video:

    python -m pipeline.step_07_merge_gps --video zurich_main
"""
