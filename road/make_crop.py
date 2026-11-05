"""Make one VLM input image: the box plus context, with the box drawn in red."""
import sys

from PIL import Image, ImageDraw


def make_crop(frame_path, bbox_xyxy, out_path, margin=0.3, min_context=48, long_side=512):
    """Crop the box with context on each side, resize the long side, then draw the box outline."""
    img = Image.open(frame_path).convert("RGB")
    x1, y1, x2, y2 = bbox_xyxy
    pad_x = max(margin * (x2 - x1), min_context)
    pad_y = max(margin * (y2 - y1), min_context)
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(img.width, int(x2 + pad_x)), min(img.height, int(y2 + pad_y))
    crop = img.crop((cx1, cy1, cx2, cy2))
    scale = long_side / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.BICUBIC)
    box = [(x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale]
    ImageDraw.Draw(crop).rectangle(box, outline=(229, 50, 45), width=3)
    crop.save(out_path, quality=92)
    return out_path


if __name__ == "__main__":
    # Usage: python make_crop.py FRAME.jpg x1 y1 x2 y2 OUT.jpg
    frame, x1, y1, x2, y2, out = sys.argv[1:7]
    print(make_crop(frame, [int(x1), int(y1), int(x2), int(y2)], out))
