"""Write Label Studio tasks from the Detector's boxes. One task per box.

No sentences are written here: road-captions writes the sentence live, when the intern presses a key.

Input: JSONL with one box per line:
  {"record_id": "...", "crop": "s3://bucket/crops/x.jpg", "frame": "s3://bucket/frames/y.jpg",
   "detector_label": "crack", "detector_score": 0.41}

Usage:
  python prepare_tasks.py --boxes boxes.jsonl --out tasks.json
"""
import argparse
import json

from road_captions.sentences import DEFECTS, DETECTOR_LOOKALIKES

KNOWN = set(DEFECTS) | set(DETECTOR_LOOKALIKES)


def build_task(box):
    det = box["detector_label"]
    score = box.get("detector_score")
    proposal = f"Detector: {det}" + (f" ({score:.2f})" if isinstance(score, (int, float)) else "")
    return {"data": {"record_id": box["record_id"], "crop": box["crop"], "frame": box.get("frame", box["crop"]),
                     "detector_label": det, "proposal_text": proposal}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", required=True, help="JSONL, one box per line")
    ap.add_argument("--out", required=True, help="Label Studio tasks JSON to write")
    args = ap.parse_args()

    boxes = [json.loads(line) for line in open(args.boxes, encoding="utf-8") if line.strip()]
    unknown = sorted({b["detector_label"] for b in boxes} - KNOWN)
    if unknown:
        raise SystemExit(f"unknown detector labels {unknown}; known: {sorted(KNOWN)}")
    ids = [b["record_id"] for b in boxes]
    if len(ids) != len(set(ids)):
        raise SystemExit("record_id values are not unique")
    tasks = [build_task(b) for b in boxes]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)
    print(f"boxes {len(boxes)}, tasks written {len(tasks)} -> {args.out}")


if __name__ == "__main__":
    main()
