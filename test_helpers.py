import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_image import annotate


def test_annotate_creates_output(tmp_path):
    import cv2
    import numpy as np

    image_path = tmp_path / "input.jpg"
    output_path = tmp_path / "output.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))

    annotate(image_path, {"predictions": []}, output_path)

    assert output_path.exists()
    assert cv2.imread(str(output_path)) is not None
