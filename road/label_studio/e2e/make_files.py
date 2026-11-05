"""Make synthetic road crops and frames plus tasks.json for the end-to-end test."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BOXES = [("t1_pothole", "pothole", 0.47), ("t2_manhole", "manhole", 0.88), ("t3_crack", "crack", 0.41),
         ("t4_raveling", "raveling", 0.58), ("t5_badbox", "raveling", 0.33)]


def asphalt(w, h, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), (92, 92, 96))
    px = img.load()
    for _ in range(w * h // 3):
        x, y, v = rnd.randrange(w), rnd.randrange(h), rnd.randrange(60, 130)
        px[x, y] = (v, v, v + 4)
    return img


def main():
    (HERE / "files" / "crops").mkdir(parents=True, exist_ok=True)
    (HERE / "files" / "frames").mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, (rid, label, score) in enumerate(BOXES):
        crop = asphalt(512, 358, i)
        d = ImageDraw.Draw(crop)
        if label == "manhole":
            d.ellipse((186, 109, 326, 249), fill=(55, 55, 60), outline=(150, 150, 150), width=6)
        elif label == "crack":
            d.line([(60, 300), (200, 190), (330, 170), (460, 60)], fill=(30, 30, 30), width=5)
        else:
            d.ellipse((170, 110, 350, 250), fill=(45, 45, 50))
        d.rectangle((150, 90, 370, 270), outline=(229, 50, 45), width=3)
        crop.save(HERE / "files" / "crops" / f"{rid}.jpg", quality=88)
        frame = asphalt(960, 540, 100 + i)
        frame.paste(crop.resize((256, 179)), (352, 300))
        frame.save(HERE / "files" / "frames" / f"{rid}.jpg", quality=85)
        tasks.append({"data": {"record_id": rid, "crop": f"/data/local-files/?d=e2e/crops/{rid}.jpg",
                               "frame": f"/data/local-files/?d=e2e/frames/{rid}.jpg", "detector_label": label,
                               "proposal_text": f"Detector: {label} ({score:.2f})"}})
    (HERE / "tasks.json").write_text(json.dumps(tasks, indent=1))
    print(f"wrote {len(tasks)} tasks and images to {HERE / 'files'}")


if __name__ == "__main__":
    main()
