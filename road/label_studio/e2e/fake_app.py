"""road-captions with a fake model, for the end-to-end test only. It never calls Bedrock."""
import re
import time

from road_captions.app import Settings, create_app

FROM_IMAGE = {
    "pothole": "Pothole with broken edges and water inside.",
    "crack": "Crack running along the lane.",
    "raveling": "Raveling: loose stones across the lane.",
    "no defect": "No defect: oil stain on smooth asphalt.",
}


class FakeModel:
    def write(self, prompt, image=None, image_format=None):
        time.sleep(0.8)  # about as slow as a real call
        words = re.search(r'Reviewer\'s words: "(.*)"', prompt)
        if words:
            label = re.search(r"Label \(fixed, do not change it\): (.+)", prompt).group(1).strip()
            head = "No defect:" if label == "no defect" else label.capitalize() + " with"
            return f"{head} {words.group(1)}."
        if image is None or image_format != "jpeg":
            raise RuntimeError("expected the crop as JPEG bytes")
        return FROM_IMAGE[re.search(r"label of the red box: (.+?)\.\n", prompt).group(1)]


def create():
    return create_app(Settings.from_env(), writer=FakeModel())
