# Helmet Detection

A computer-vision project for detecting helmet-related objects in images and video.

## Work completed

- Explored and prepared a helmet-detection dataset using Roboflow.
- Used a Roboflow-hosted **RF-DETR Object Detection (Small)** model.
- Tested the model in Roboflow's interface.
- Recorded the displayed evaluation metrics:
  - mAP@50: **97.7%**
  - Precision: **89.1%**
  - Recall: **98.0%**
- Tested an image without a helmet. In that test, the model returned **"No object detected"**. This is an observed test result, not proof that the model works reliably in every situation.

## Important: model and dataset are not bundled

The original trained weights and dataset files were not available when this archive was assembled, so they are not fabricated or replaced with dummy files. The project code is ready to connect to your Roboflow model once you add your own API key and model ID.

The model identifier shown during testing was `helmet-kwmhk-knbzp`. The exact version identifier and API key are account-specific and must be supplied by you.

## Setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`.
5. Add your Roboflow API key and the exact model ID shown in your Roboflow Deploy page.

Example `.env`:

```text
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_MODEL_ID=your-project-slug/1
```

Do not commit `.env` or your API key to GitHub.

## Run image detection

```bash
python src/predict_image.py path/to/image.jpg --output outputs/prediction.jpg
```

## Run video detection

```bash
python src/predict_video.py path/to/video.mp4 --output outputs/prediction.mp4
```

Video inference processes frames one by one. It can be slower than real-time depending on the machine and hosted inference latency.

## Project structure

```text
Helmet-Detection-Project/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── roboflow_client.py
│   ├── predict_image.py
│   └── predict_video.py
├── config/
│   └── model_config.example.yaml
├── docs/
│   ├── project_notes.md
│   └── results_and_limitations.md
├── tests/
│   └── test_helpers.py
└── outputs/
    └── README.md
```

## Honest reporting

The metrics above are the values displayed by Roboflow for the selected model. They should not be described as independently reproduced results unless you rerun evaluation on the same held-out test set. A single image test that returns no detection does not establish whether the image is correctly classified; inspect the image, confidence threshold, labels, and model response.

## Next steps

- Add the exact model ID from Roboflow.
- Run inference on several helmet and no-helmet images.
- Save genuine prediction outputs in `outputs/`.
- Review false positives and false negatives.
- If publishing the dataset or model weights, check the dataset/model license and repository size limits first.
