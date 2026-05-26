"""Sanity checks that the model-side scripts are actually runnable.

These tests do NOT touch the GPU or the Modal API; they verify the
*data contracts* between derive_variants.py → train_modal.py → eval_modal.py.
A bug in any one of those shapes is silent until $20-of-Modal-GPU later,
so it's worth pinning here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import derive_variants as dv          # noqa: E402
from src import eval_metrics as em             # noqa: E402


# ── a realistic phase-B annotation record ────────────────────────────
SAMPLE = {
    "video": "bahnhofstrasse",
    "frame_id": "frame_00123",
    "gps": [47.37498, 8.53696],
    "heading": 220.0,
    "dest_name": "Grossmünster",
    "dest_gps": [47.37, 8.544],
    "dest_dist_m": 400.0,
    "route_bearing": 130.0,
    "route_distance_m": 480.0,
    "route_latlon": [[47.37498, 8.53696], [47.37, 8.544]],
    "nearby_pois": ["Bahnhofstrasse", "St. Peter", "Münsterhof"],
    "thinking": "STEP 1 SCENE: street\nSTEP 3 INFERRED_HEADING: 220\n"
                "STEP 5 ACTION: turn left\n",
    "answer": "Turn left at the tram tracks. Walk a few blocks "
              "towards the spires until you reach Grossmünster.",
    "accepted": True,
}


# ─── derive_variants → train data shape ──────────────────────────────
def test_derive_variants_message_structure():
    row = dv.to_message_row(SAMPLE, "given")
    assert row["image_rel"] == "bahnhofstrasse/frame_00123.jpg"
    msgs = row["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    # critical: the user content MUST contain the image placeholder
    user_content = msgs[1]["content"]
    types = [p.get("type") for p in user_content]
    assert "image" in types, ("no image placeholder — Qwen2.5-VL will "
                              "train text-only")
    assert "text" in types
    # assistant content is structured too
    assert msgs[2]["content"][0]["type"] == "text"


def test_given_keeps_heading_line_and_step():
    row = dv.to_message_row(SAMPLE, "given")
    user_text = row["messages"][1]["content"][1]["text"]
    assistant = row["messages"][2]["content"][0]["text"]
    assert "My camera heading:" in user_text
    assert "INFERRED_HEADING" in assistant


def test_implicit_strips_both_heading_and_step():
    row = dv.to_message_row(SAMPLE, "implicit")
    user_text = row["messages"][1]["content"][1]["text"]
    assistant = row["messages"][2]["content"][0]["text"]
    assert "My camera heading:" not in user_text
    assert "INFERRED_HEADING" not in assistant


def test_explicit_strips_user_but_keeps_step():
    row = dv.to_message_row(SAMPLE, "explicit")
    user_text = row["messages"][1]["content"][1]["text"]
    assistant = row["messages"][2]["content"][0]["text"]
    assert "My camera heading:" not in user_text          # hidden from user
    assert "INFERRED_HEADING" in assistant                # but kept in CoT


def test_all_variants_have_distinct_system_prompts():
    sys_msgs = [
        dv.to_message_row(SAMPLE, v)["messages"][0]["content"][0]["text"]
        for v in ("given", "implicit", "explicit")
    ]
    assert len({s for s in sys_msgs}) == 3                # all different


# ─── eval_metrics PASS_strict logic ──────────────────────────────────
def test_format_compliance_rejects_compass_words():
    raw = ("<thinking>...</thinking><answer>"
           "Walk north for a block, then turn right.</answer>")
    ok, reasons = em.format_compliance(raw, "Walk north for a block, "
                                             "then turn right.")
    assert ok is False
    assert any("compass" in r for r in reasons)


def test_format_compliance_passes_clean_answer():
    answer = "Turn left at the tram tracks. Walk towards the spires."
    raw = f"<thinking>...</thinking><answer>{answer}</answer>"
    ok, _ = em.format_compliance(raw, answer)
    assert ok is True


def test_directional_accuracy_correct_action():
    # heading 0 (north), route bearing 90 (east) → 'turn right' is correct
    ok, delta = em.directional_accuracy(0, "turn right", 90)
    assert ok is True
    assert delta == 0


def test_directional_accuracy_wrong_action():
    ok, delta = em.directional_accuracy(0, "turn left", 90)
    assert ok is False
    assert abs(delta - 180) < 1


def test_parse_action_verb_finds_first():
    assert em.parse_action_verb("Turn LEFT at the tracks.") == "turn left"
    assert em.parse_action_verb("Just continue ahead a few blocks.") \
        == "continue ahead"
    assert em.parse_action_verb("Nothing actionable here.") is None


def test_extract_anchor():
    a = em.extract_anchor("Turn left at the tram tracks. Walk on.")
    assert a == "tram tracks"


def test_extract_checkpoint():
    c = em.extract_checkpoint("Walk on. When you reach Grossmünster, "
                              "send another photo.")
    assert c == "Grossmünster"


def test_anchor_faithfulness_no_anchor_passes():
    ok, raw = em.anchor_faithfulness("/nonexistent.jpg", None)
    assert ok is True
    assert raw == ""


def test_anchor_faithfulness_injected_caller():
    # pretend Gemini said YES
    ok, raw = em.anchor_faithfulness(
        "/img.jpg", "tram tracks",
        gemini_caller=lambda *a, **kw: "YES")
    assert ok is True
    # pretend Gemini said NO
    ok2, _ = em.anchor_faithfulness(
        "/img.jpg", "unicorn",
        gemini_caller=lambda *a, **kw: "no")
    assert ok2 is False


def test_pass_strict_is_and():
    assert em.pass_strict(True, True, True, True) is True
    assert em.pass_strict(True, True, True, False) is False
    assert em.pass_strict(False, True, True, True) is False


# ─── eval_modal contract ─────────────────────────────────────────────
def test_eval_modal_messages_match_train_shape():
    """The most important test: the messages eval_modal builds at
    inference time MUST have the same shape (image placeholder + text)
    as the ones the LoRA was trained on. A mismatch silently breaks
    the L-* conditions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "em_mod", Path(__file__).resolve().parent.parent
        / "eval_modal.py")
    em_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em_mod)

    sample = dict(SAMPLE)
    sample["image_rel"] = "bahnhofstrasse/frame_00123.jpg"
    for variant in ("given", "implicit", "explicit"):
        messages, img_path = em_mod.build_eval_messages(sample, variant)
        assert [m["role"] for m in messages] == ["system", "user"]
        types = [p.get("type") for p in messages[1]["content"]]
        assert "image" in types, f"{variant}: missing image placeholder"
        assert "text" in types

        # the user-text must be identical to what derive_variants
        # wrote (modulo trailing newlines)
        train_row = dv.to_message_row(sample, variant)
        train_user = train_row["messages"][1]["content"][1]["text"]
        eval_user = messages[1]["content"][1]["text"]
        assert train_user.strip() == eval_user.strip(), variant


def test_eval_modal_system_prompts_match_derive():
    """The system prompts must match too — the L-* LoRA was trained
    under one prompt; if eval uses a different one, the adapter is
    half-blind."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "em_mod", Path(__file__).resolve().parent.parent
        / "eval_modal.py")
    em_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em_mod)
    for v in ("given", "implicit", "explicit"):
        assert em_mod.SYS_PROMPTS[v] == dv.SYS_PROMPTS[v], v
