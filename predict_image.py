"""Run hosted Roboflow inference on one image and save an annotated copy."""
import argparse
import json
from pathlib import Path

import cv2

from roboflow_client import infer_image


def annotate(image_path, prediction, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for item in prediction.get("predictions", []):
        x, y = float(item["x"]), float(item["y"])
        w, h = float(item["width"]), float(item["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = str(item.get("class", "object"))
        confidence = float(item.get("confidence", 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            image, f"{label} {confidence:.2f}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save output image: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output", default="outputs/prediction.jpg")
    parser.add_argument("--json", default="outputs/prediction.json")
    args = parser.parse_args()

    result = infer_image(args.image)
    saved = annotate(args.image, result, args.output)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    detections = result.get("predictions", [])
    print(f"Detections: {len(detections)}")
    print(f"Annotated image: {saved}")
    print(f"Raw response: {json_path}")


if __name__ == "__main__":
    main()
