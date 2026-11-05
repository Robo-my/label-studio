"""GRPO reward for the road-damage VLM.

The model answers each record with JSON: {"label": ..., "caption": ...}.
Only the label is compared with the record's "solution". The caption is
checked by rules, never against a stored sentence.
"""
import json
import re

LABELS = {"pothole", "crack", "raveling", "no_defect"}
BANNED = {"severe", "dangerous", "minor", "major", "large", "small",
          "deep", "shallow", "huge", "tiny"}
STEMS = {"pothole": "pothole", "crack": "crack", "raveling": "ravel"}
MAX_WORDS = 15


def defects_named(text):
    """Defect labels that the text names. A word must start with the stem, so 'gravel' is not 'raveling'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {label for label, stem in STEMS.items() if any(w.startswith(stem) for w in words)}


def caption_ok(caption, label):
    """True when the caption follows the caption rules for its label."""
    text = caption.strip().lower()
    words = re.findall(r"[a-z0-9]+", text)
    if not words or len(words) > MAX_WORDS:
        return False
    if any(w in BANNED for w in words) or re.search(r"\d", text):
        return False
    if label == "no_defect":  # a no-defect sentence must not name a defect
        return text.startswith("no defect") and not defects_named(text)
    return defects_named(text) == {label}


def reward(completion, solution):
    """Score one answer: 0.2 valid JSON + 0.8 correct label - 0.2 if a caption rule breaks."""
    try:
        out = json.loads(completion)
    except (TypeError, ValueError):
        return 0.0
    if not isinstance(out, dict) or out.get("label") not in LABELS:
        return 0.0
    score = 0.2
    if out["label"] == solution:
        score += 0.8
    if not caption_ok(str(out.get("caption", "")), out["label"]):
        score -= 0.2
    return max(0.0, score)


def grpo_reward(completions, solution, **kwargs):
    """Batch form for trainers that pass completions plus dataset columns as keyword arguments."""
    texts = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    return [reward(t, s) for t, s in zip(texts, solution)]


def uncertainty(rewards):
    """WeVisOne-VL selection score: U = 4 * mean * (1 - mean) * std. Zero means the record teaches nothing."""
    n = len(rewards)
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
    return 4 * mean * (1 - mean) * std


if __name__ == "__main__":
    checks = [
        ('{"label": "raveling", "caption": "Raveling: loose stones and rough surface."}', "raveling", 1.0),
        ('{"label": "crack", "caption": "Single crack running along the lane."}', "raveling", 0.2),
        ('{"label": "raveling", "caption": "Severe raveling, dangerous for motorcycles."}', "raveling", 0.8),
        ('{"label": "pothole", "caption": "Pothole next to a long crack."}', "pothole", 0.8),
        ('{"label": "no_defect", "caption": "No defect: manhole cover on intact asphalt."}', "no_defect", 1.0),
        ("raveling, I think", "raveling", 0.0),
    ]
    for completion, solution, expected in checks:
        got = reward(completion, solution)
        status = "ok " if abs(got - expected) < 1e-9 else "BAD"
        print(f"{status} {got:.1f}  answer={completion}  correct={solution}")
    print("U with 4 of 8 right:", round(uncertainty([1.0] * 4 + [0.2] * 4), 3))
    print("U with 8 of 8 right:", round(uncertainty([1.0] * 8), 3))
