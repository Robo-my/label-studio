"""Turn Label Studio JSON exports into verified records, SFT and GRPO files, and a QA report.

Usage:
  python ls_export_to_jsonl.py --exports export.json --golden golden_answers.json --out out --grpo-share 0.2

- The sentence is saved live by road-captions in the hidden "caption" field; the converter checks it again
  with the same caption rules (reward.py) and, when the intern typed words, that it keeps every word.
- golden_answers.json maps record_id to the correct label for the hidden golden check tasks.
  Keep these answers out of the task data, so interns cannot see them.
- Records seen by two interns count as overlap; a disagreement goes to the QA queue.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from road_captions.sentences import check, resolve_label

PROMPT = "<image>Is there a road defect inside the red box? Answer as JSON with label and caption."


def read_answers(annotation):
    """Collect the intern's choices and texts from one Label Studio annotation."""
    out = {}
    for item in annotation.get("result", []):
        name, value = item.get("from_name"), item.get("value", {})
        if value.get("choices"):
            out[name] = value["choices"][0]
        elif value.get("text"):
            out[name] = " ".join(value["text"]).strip()
    return out


def resolve(data, answers):
    """Return (status, label, caption, reason). Status: verified, qa or dropped."""
    decision = answers.get("decision")
    if decision is None:
        return "qa", None, None, "no decision"
    if decision == "bad_box":
        return "dropped", None, None, "bad box"
    if decision == "unsure":
        return "qa", None, None, "unsure"
    try:
        label, _ = resolve_label(decision, answers.get("lookalike"), data.get("detector_label", ""))
    except ValueError as exc:
        return "qa", None, None, str(exc)
    caption = answers.get("caption", "")
    if not caption:
        return "qa", label, None, "no sentence saved"
    problem = check(caption, label, answers.get("words", ""))
    if problem:
        return "qa", label, caption, f"sentence {problem}"
    return "verified", label, caption, ""


def intern_name(annotation):
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        who = who.get("email") or who.get("id")
    return str(who)


def sft_record(rec):
    answer = json.dumps({"label": rec["label"], "caption": rec["caption"]})
    return {"messages": [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": answer}],
            "images": [rec["crop"]],
            "meta": {"id": rec["record_id"], "interns": rec["interns"], "caption_source": rec["caption_source"],
                     "split": "sft"}}


def grpo_record(rec):
    return {"messages": [{"role": "user", "content": PROMPT}], "images": [rec["crop"]],
            "solution": rec["label"], "meta": {"id": rec["record_id"], "interns": rec["interns"],
                                               "split": "grpo_candidate"}}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", nargs="+", required=True, help="Label Studio JSON exports")
    ap.add_argument("--golden", help="JSON file: record_id -> correct label, for hidden check tasks")
    ap.add_argument("--out", default="out", help="output folder")
    ap.add_argument("--grpo-share", type=float, default=0.2, help="share of verified records held back for GRPO, per class")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    golden = json.load(open(args.golden, encoding="utf-8")) if args.golden else {}
    answers = defaultdict(list)
    for path in args.exports:
        for task in json.load(open(path, encoding="utf-8")):
            data = task["data"]
            for ann in task.get("annotations", []):
                if ann.get("was_cancelled"):
                    continue
                ans = read_answers(ann)
                status, label, caption, reason = resolve(data, ans)
                answers[data["record_id"]].append({
                    "intern": intern_name(ann), "status": status, "label": label, "caption": caption,
                    "reason": reason, "data": data, "words": ans.get("words", ""),
                    "caption_source": ans.get("caption_source", "")})

    golden_score = defaultdict(lambda: [0, 0])  # intern -> [right, checked]
    done = Counter()
    overlap_pairs = overlap_agree = 0
    final = {"verified": [], "qa": [], "dropped": []}
    for rid, items in answers.items():
        for a in items:
            done[a["intern"]] += 1
        if rid in golden:  # hidden check task: score it, never train on it
            for a in items:
                golden_score[a["intern"]][1] += 1
                golden_score[a["intern"]][0] += int(a["label"] == golden[rid])
            continue
        first = items[0]
        rec = {"record_id": rid, "crop": first["data"].get("crop"), "label": first["label"],
               "caption": first["caption"], "caption_source": first["caption_source"], "words": first["words"],
               "interns": sorted({a["intern"] for a in items}), "reason": first["reason"]}
        status = first["status"]
        if len({a["intern"] for a in items}) > 1:
            overlap_pairs += 1
            if len({(a["status"], a["label"]) for a in items}) == 1:
                overlap_agree += 1
            else:
                status = "qa"
                rec["reason"] = "interns disagree: " + ", ".join(
                    f"{a['intern']}={a['label'] or a['status']}" for a in items)
        final[status].append(rec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in final.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    rng = random.Random(args.seed)
    by_class = defaultdict(list)
    for rec in final["verified"]:
        by_class[rec["label"]].append(rec)
    sft, grpo = [], []
    for label in sorted(by_class):
        recs = by_class[label]
        rng.shuffle(recs)
        cut = int(len(recs) * args.grpo_share)
        grpo += recs[:cut]
        sft += recs[cut:]
    write_jsonl(out / "sft.jsonl", [sft_record(r) for r in sft])
    write_jsonl(out / "grpo_candidates.jsonl", [grpo_record(r) for r in grpo])

    print("records:", {k: len(v) for k, v in final.items()})
    print("verified per class:", dict(Counter(r["label"] for r in final["verified"])))
    print("sentence source:", dict(Counter(r["caption_source"] or "unknown" for r in final["verified"])))
    print(f"SFT {len(sft)}, GRPO candidates {len(grpo)}")
    if overlap_pairs:
        print(f"overlap: {overlap_agree}/{overlap_pairs} records agree ({100 * overlap_agree / overlap_pairs:.0f}%)")
    for intern in sorted(done):
        right, checked = golden_score[intern]
        acc = f"{right}/{checked} golden right" if checked else "no golden checks yet"
        print(f"  {intern}: {done[intern]} answers, {acc}")
    print("files written to", out.resolve())


if __name__ == "__main__":
    main()
