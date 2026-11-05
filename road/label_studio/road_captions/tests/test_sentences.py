import pytest

from road_captions.sentences import (FALLBACK, LOOKALIKE_CAPTIONS, caption_prompt, check, clean_reply,
                                     intern_sentence, keeps_words, resolve_label, rewrite_prompt,
                                     template_sentence, words_problem)


@pytest.mark.parametrize("decision, lookalike, detector, expected", [
    ("Correct", None, "pothole", ("pothole", None)),
    ("Correct", None, "manhole", ("no_defect", "manhole")),
    ("Correct", None, "stagnant_water", ("no_defect", "standing_water")),
    ("crack", None, "pothole", ("crack", None)),
    ("not_a_defect", "shadow", "crack", ("no_defect", "shadow")),
    ("not_a_defect", None, "crack", ("no_defect", None)),
    ("bad_box", None, "crack", (None, None)),
    ("unsure", None, "crack", (None, None)),
])
def test_resolve_label(decision, lookalike, detector, expected):
    assert resolve_label(decision, lookalike, detector) == expected


def test_resolve_label_rejects_unknown_values():
    with pytest.raises(ValueError):
        resolve_label("Correct", None, "speed_bump")
    with pytest.raises(ValueError):
        resolve_label("maybe", None, "pothole")


def test_fixed_sentences_pass_the_rules():
    for label, sentence in FALLBACK.items():
        assert check(sentence, label) == ""
    for sentence in LOOKALIKE_CAPTIONS.values():
        assert check(sentence, "no_defect") == ""


@pytest.mark.parametrize("label, words, expected", [
    ("pothole", "water inside, broken edges", "Pothole: water inside, broken edges."),
    ("pothole", "potholes filled with water", "Potholes filled with water."),
    ("crack", "long line along lane", "Crack: long line along lane."),
    ("no_defect", "oil stain", "No defect: oil stain."),
    ("raveling", "deep loose stones", "Raveling: loose stones."),  # severity words are dropped
    ("pothole", "pothole", "Pothole in the road surface."),       # only the label word: fixed sentence
])
def test_template_sentence(label, words, expected):
    assert template_sentence(label, words) == expected


def test_template_sentences_keep_the_words():
    for label, words in [("pothole", "water inside, broken edges"), ("no_defect", "oil stain on lane")]:
        sentence = template_sentence(label, words)
        assert check(sentence, label, words) == ""


def test_intern_sentence_used_as_typed_only_when_valid():
    assert intern_sentence("pothole", "pothole with water inside") == "Pothole with water inside."
    assert intern_sentence("pothole", "water inside") == ""          # no label word
    assert intern_sentence("crack", "crack next to a pothole") == ""  # names another defect


def test_keeps_words_folds_plurals():
    assert keeps_words("Pothole with broken edge.", "broken edges")
    assert keeps_words("Raveling: loose stones in patches.", "loose stone, patch")
    assert keeps_words("Crack with one cavity.", "cavities")
    assert not keeps_words("Pothole with broken rim.", "broken edges")
    assert not keeps_words("Crack across the glass.", "grass")


def test_words_problem():
    assert "pothole" in words_problem("crack", "pothole edge")
    assert "a defect" in words_problem("no_defect", "small crack")
    assert words_problem("pothole", "water inside, broken edges") == ""
    assert words_problem("pothole", " ".join(["word"] * 13)) == "Use 2-6 words."


def test_prompts_name_the_label_and_the_other_defects():
    prompt = caption_prompt("crack")
    assert 'include the word "crack"' in prompt and "pothole, raveling" in prompt
    prompt = rewrite_prompt("no_defect", 'oil "stain"')
    assert 'start with "No defect:"' in prompt and "oil 'stain'" in prompt


@pytest.mark.parametrize("reply, expected", [
    ("Pothole with broken edges.", "Pothole with broken edges."),
    ('"Pothole with broken edges."\n\nThis follows the rules.', "Pothole with broken edges."),
    ("Sentence: Crack along the lane.", "Crack along the lane."),
    ("\n\n", ""),
])
def test_clean_reply(reply, expected):
    assert clean_reply(reply) == expected
