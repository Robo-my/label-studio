"""Sentence logic for the review screen: labels, fixed sentences, prompts and checks.

No network calls here. The road-captions service, the export converter and the tests share this file.
The caption rules themselves live in reward.py, so review and GRPO training use the same rules.
"""
import re
import sys
from pathlib import Path

try:
    from reward import BANNED, MAX_WORDS, caption_ok, defects_named
except ImportError:  # in the repo, reward.py is two folders up
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from reward import BANNED, MAX_WORDS, caption_ok, defects_named

DEFECTS = ("pothole", "crack", "raveling")
LABEL_WORD = {"pothole": "Pothole", "crack": "Crack", "raveling": "Raveling", "no_defect": "No defect"}
FALLBACK = {
    "pothole": "Pothole in the road surface.",
    "crack": "Crack in the road surface.",
    "raveling": "Raveling: loose material on the road surface.",
}
LOOKALIKE_CAPTIONS = {
    "manhole": "No defect: manhole cover on intact asphalt.",
    "standing_water": "No defect: standing water on intact asphalt.",
    "shadow": "No defect: shadow on smooth asphalt.",
    "patch": "No defect: patched area in good condition.",
    "road_marking": "No defect: road marking.",
    "intact_asphalt": "No defect: intact asphalt.",
}
LOOKALIKES = set(LOOKALIKE_CAPTIONS) | {"other"}
DETECTOR_LOOKALIKES = {"manhole": "manhole", "stagnant_water": "standing_water"}
NO_SENTENCE = {"bad_box", "unsure"}
DECISIONS = {"Correct", *DEFECTS, "not_a_defect", *NO_SENTENCE}
MAX_TYPED_WORDS = 12  # the screen asks for 2-6 words; allow some slack
STOP = {"a", "an", "the", "of", "with", "and", "or", "on", "in", "inside", "at", "to", "for", "is", "are",
        "be", "it", "its", "there", "this", "that", "some", "has", "have", "from", "by", "very", "no", "defect"}

CAPTION_PROMPT = (
    "The image is a crop from a dashcam road photo. The red box marks one area.\n"
    "A reviewer confirmed the label of the red box: {label}.\n"
    "Write one sentence about what is visible inside the red box, for a training dataset.\n"
    "Rules: {start_rule}; do not name any other defect type ({others}); no severity or size words "
    "(severe, minor, major, large, small, deep, shallow, dangerous, huge, tiny); no numbers; "
    "at most 15 words.\n"
    "Answer with the sentence only."
)
REWRITE_PROMPT = (
    "Write one sentence about a road photo for a training dataset.\n"
    "Label (fixed, do not change it): {label}\n"
    'Reviewer\'s words: "{words}"\n'
    "Rules: keep every reviewer word and its meaning; add only small linking words; {start_rule}; "
    "do not name any other defect type ({others}); no severity or size words (severe, minor, major, "
    "large, small, deep, shallow, dangerous, huge, tiny); no numbers; at most 15 words.\n"
    "Answer with the sentence only."
)


def resolve_label(decision, lookalike, detector_label):
    """Return (label, lookalike) for one decision. label is None for bad_box and unsure (no sentence)."""
    if decision in NO_SENTENCE:
        return None, None
    if decision == "Correct":
        if detector_label in DEFECTS:
            return detector_label, None
        if detector_label in DETECTOR_LOOKALIKES:
            return "no_defect", DETECTOR_LOOKALIKES[detector_label]
        raise ValueError(f"unknown detector label {detector_label!r}")
    if decision in DEFECTS:
        return decision, None
    if decision == "not_a_defect":
        return "no_defect", (lookalike or None)
    raise ValueError(f"unknown decision {decision!r}")


def tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def norm(word):
    """Fold simple plurals: edges -> edge, patches -> patch, cavities -> cavity."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def clean_words(words):
    """The intern's words, one line, no extra spaces."""
    return re.sub(r"\s+", " ", (words or "").replace("\n", " ")).strip()


def required_words(words):
    """Content words a new sentence must keep. Severity and size words and numbers are dropped on purpose."""
    return {norm(w) for w in tokens(words)
            if w not in STOP and w not in BANNED and not re.search(r"\d", w)}


def keeps_words(sentence, words):
    return required_words(words) <= {norm(w) for w in tokens(sentence)}


def as_sentence(text):
    """Capital first letter, one final full stop."""
    text = clean_words(text).rstrip(" .")
    return (text[:1].upper() + text[1:] + ".") if text else ""


def intern_sentence(label, words):
    """The intern's words used as they are, when they already make a valid sentence."""
    sentence = as_sentence(words)
    return sentence if caption_ok(sentence, label) else ""


def template_sentence(label, words):
    """A sentence built from the intern's words without a model: 'Pothole: water inside, broken edges.'"""
    chunks = []
    for chunk in re.split(r"[,;/]+", words or ""):
        kept = [w for w in tokens(chunk) if w not in BANNED and not re.search(r"\d", w)]
        if kept:
            chunks.append(" ".join(kept))
    phrase = ", ".join(chunks)
    label_words = {norm(w) for w in tokens(LABEL_WORD.get(label, ""))}
    if not phrase or {norm(w) for w in tokens(phrase)} <= label_words:
        return FALLBACK.get(label, "")
    if label == "no_defect":
        starts_with_label = phrase.lower().startswith("no defect")
    else:
        starts_with_label = defects_named(tokens(phrase)[0]) == {label}
    sentence = as_sentence(phrase) if starts_with_label else f"{LABEL_WORD[label]}: {phrase}."
    words_out = sentence.split()
    return sentence if len(words_out) <= MAX_WORDS else " ".join(words_out[:MAX_WORDS]).rstrip(",") + "."


def check(sentence, label, words=""):
    """Return '' when the sentence can be saved, else the problem."""
    if not sentence:
        return "empty sentence"
    if not caption_ok(sentence, label):
        return "breaks the caption rules"
    if words and not keeps_words(sentence, words):
        return "drops or changes the intern's words"
    return ""


def words_problem(label, words):
    """A short message for the intern when their words cannot make a valid sentence, else ''."""
    if len(tokens(words)) > MAX_TYPED_WORDS:
        return "Use 2-6 words."
    named = sorted(defects_named(words) - ({label} if label != "no_defect" else set()))
    if named:
        what = "a defect" if label == "no_defect" else "another defect"
        return f"Your words name {what} ({', '.join(named)}). Change the words, or change the key."
    return ""


def start_rule(label):
    return 'start with "No defect:"' if label == "no_defect" else f'include the word "{label}"'


def others(label):
    return ", ".join(d for d in DEFECTS if d != label)


def caption_prompt(label):
    return CAPTION_PROMPT.format(label=label.replace("_", " "), start_rule=start_rule(label), others=others(label))


def rewrite_prompt(label, words):
    return REWRITE_PROMPT.format(label=label.replace("_", " "), words=words.replace('"', "'"),
                                 start_rule=start_rule(label), others=others(label))


def clean_reply(text):
    """First non-empty line of a model reply, without quotes or a 'Sentence:' prefix."""
    for line in (text or "").splitlines():
        line = line.strip().strip('"`*').strip()
        line = re.sub(r"^(sentence|answer)\s*:\s*", "", line, flags=re.I).strip('"` ')
        if line:
            return line
    return ""
