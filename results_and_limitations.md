# Results and limitations

## Metrics displayed in Roboflow

| Metric | Displayed value |
|---|---:|
| mAP@50 | 97.7% |
| Precision | 89.1% |
| Recall | 98.0% |

These are transcribed from the Roboflow test page shown during project work. They are not independently recalculated in this repository.

## Observed manual test

An image without a helmet was uploaded for testing, and the interface returned **No object detected**. This is a useful test observation, but it is not enough to conclude that the project is complete or that the model correctly identifies every no-helmet case.

## What still needs validation

- Test several images containing helmets.
- Test several images showing people without helmets.
- Check that the model detects people/heads if those are intended classes.
- Review confidence thresholds and class labels.
- Record false positives and false negatives.
- Test video footage with different lighting, angles, distances, and occlusions.

A high mAP score on a held-out test set does not guarantee reliable performance on every real-world image.
