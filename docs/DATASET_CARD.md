# Dataset Card — Calculator Segmentation Dataset

## Overview

This dataset was collected and labelled manually for the purpose of
training an instance segmentation model to detect and measure a
Casio fx-991ES PLUS scientific calculator using a calibrated camera pipeline.

---

## Object Description

| Property | Details |
|---|---|
| Object | Casio fx-991ES PLUS Scientific Calculator |
| Real Width | 77.0 mm |
| Real Height | 161.5 mm |
| Color | White / Grey body with orange and blue keys |
| Shape | Rectangular with rounded corners |
| Surface | Matte plastic — minimal glare |

---

## Why This Object Was Chosen

- Clearly defined rectangular boundary — easy to annotate precisely
- Rigid object — shape does not deform across images
- Consistent real-world dimensions — known ground truth (77mm × 161.5mm)
- Available in various orientations — portrait and slight angles
- Suitable for pixel-to-mm measurement validation

---

## Data Collection Strategy

All images were captured using the same smartphone camera used for
camera calibration (ensuring calibration parameters apply correctly).

### Variation Introduced

| Variation Type | Details |
|---|---|
| Backgrounds | Plain table, fabric, paper, floor, outdoor concrete, coloured surfaces |
| Lighting | Indoor artificial light, natural daylight, direct sunlight, mixed |
| Angles | Flat/top-down, slight left tilt, slight right tilt, mild perspective |
| Distance | Close (~30cm), medium (~50cm), far (~70cm) |
| Orientation | Portrait (upright), landscape (rotated 90°) |
| Placement | Center frame, off-center, near edges |

### Collection Rules

- Same camera used throughout (consistent intrinsics)
- All images captured after camera calibration was completed
- Each image contains exactly one calculator instance
- Object fully visible in every image 
- Images captured at native resolution 

---

## Dataset Statistics

| Split | Images | Percentage |
|---|---|---|
| Train | 70 | 73% |
| Validation | 15 | 16% |
| Test | 11 | 11% |
| **Total** | **96** | **100%** |

| Property | Value |
|---|---|
| Total images | 96 |
| Image resolution | 3456 × 4608 px (portrait) |
| Annotation type | Polygon segmentation mask |
| Number of classes | 1 (calculator) |
| Instances per image | 1 |
| Annotation format | COCO JSON (`_annotations.coco.json`) |

---

## Labelling Process

**Tool used:** Roboflow (web-based annotation interface)

**Annotation type:** Instance segmentation — polygon mask drawn around
the full boundary of the calculator in every image.

**Labelling guidelines followed:**
- Polygon drawn tightly around outer calculator edge
- Included the calculator body — excluded shadow and reflection
- Corners rounded to follow actual physical shape
- No bounding-box shortcuts — precise polygon per image

**Label export format:** COCO JSON with polygon segmentation

**Dataset folder structure after export:**
```
dataset/
    train/
        _annotations.coco.json
        IMG_XXXX.jpg
        ...
    valid/
        _annotations.coco.json
        IMG_XXXX.jpg
        ...
    test/
        _annotations.coco.json
        IMG_XXXX.jpg
        ...
```

---

## Calibration Images

A separate set of checkerboard images was collected for camera calibration.
These are NOT part of the segmentation dataset.

| Property | Value |
|---|---|
| Total calibration images | 22 |
| Checkerboard pattern | 7 × 9 inner corners |
| Square size | 20 mm × 20 mm |
| Successful detections | 17 out of 22 |

---

## Measurement Test Set

A separate small set of images was collected specifically for the
pixel-to-mm measurement validation in Step 3.

| Property | Value |
|---|---|
| Total measurement images | 5 |
| Reference object | 150mm ruler (Goldfish brand, 15cm) |
| Reference visible in image | Yes — ruler placed below calculator |
| Ground truth measured with | Physical ruler / known calculator dimensions |

---

## Class Distribution

| Class | Train | Validation | Test |
|---|---|---|---|
| calculator | 70 | 15 | 11 |

Single-class dataset — no class imbalance issues.

---

## Data Preprocessing

Before training, all images were preprocessed as follows:

1. **Undistortion** — `cv2.undistort()` applied using calibration parameters
2. **Resize** — resized from 3456×4608 to 512×512 for model input
3. **Normalization** — ImageNet mean/std normalization
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`

---

## Augmentations (Training Only)

| Augmentation | Probability |
|---|---|
| Horizontal flip | 0.5 |
| Vertical flip | 0.3 |
| Random rotate 90° | 0.3 |
| Random brightness/contrast | 0.3 |
| Gaussian blur | 0.2 |

Augmentations were applied using the `albumentations` library.
No augmentation was applied to validation or test sets.

---

## Limitations

- Single object class only — model trained exclusively for this calculator model
- Limited dataset size (96 images total) — may reduce generalization to
  unseen backgrounds or extreme lighting conditions
- Single camera source — calibration and images from same device only
- No occlusion scenarios 