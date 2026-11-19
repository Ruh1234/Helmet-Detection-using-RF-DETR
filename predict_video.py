"""Run hosted Roboflow inference on video frames and write an annotated video."""
import argparse
from pathlib import Path
import tempfile

import cv2

from roboflow_client import infer_image


def draw_predictions(frame, prediction):
    for item in prediction.get("predictions", []):
        x, y = float(item["x"]), float(item["y"])
        w, h = float(item["width"]), float(item["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = f'{item.get("class", "object")} {float(item.get("confidence", 0)):.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output", default="outputs/prediction.mp4")
    parser.add_argument("--every", type=int, default=1,
                        help="Run inference every Nth frame (default: every frame)")
    args = parser.parse_args()

    if args.every < 1:
        parser.error("--every must be at least 1")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    frame_index = 0
    last_prediction = {"predictions": []}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % args.every == 0:
                # The SDK accepts an image path; use a temporary frame file.
                with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
                    cv2.imwrite(temp.name, frame)
                    last_prediction = infer_image(temp.name)
            annotated = draw_predictions(frame, last_prediction)
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    print(f"Processed {frame_index} frames. Saved: {output}")


if __name__ == "__main__":
    main()
