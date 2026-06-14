# Image Segmentation and Metric Measurement Pipeline

This project implements an end-to-end computer vision pipeline that segments
a physical object from images and measures its real-world dimensions in
millimetres using a calibrated camera. The work was completed as a technical
assessment for the AI and Computer Vision department at XIS.

---

## What This System Does

The pipeline takes a photograph of a Casio fx-991ES PLUS scientific calculator,
segments it from the background using a deep learning model, and reports its
physical width and height in millimetres. Every measurement is grounded in
camera calibration, meaning lens distortion is removed before any pixel
distances are computed. A physical ruler placed in the same frame provides
the reference scale that converts pixel measurements into real-world units.

The system was built from scratch across three stages: camera calibration and
data collection, segmentation model training, and pixel-to-millimetre
measurement with accuracy validation.

---

## Repository Structure

```
Image-segmentation-and-detect-measurements/
│
├── calibration/
│   ├── images/                  # 22 checkerboard images used for calibration
│   ├── scripts/
│   │   └── calibrate.py         # Calibration script using OpenCV
│   ├── camera_matrix.npy        # Saved 3x3 intrinsic camera matrix
│   ├── dist_coeffs.npy          # Saved distortion coefficients
│   └── calibration_params.json  # Parameters in human-readable format
│
├── dataset/
│   ├── train/                   # 80 labelled training images + COCO JSON
│   ├── valid/                   # 21 labelled validation images + COCO JSON
│   └── test/                    # 11 labelled test images + COCO JSON
│
├── models/
│   ├── configs/
│   │   └── unet_config.json     # Training configuration and metrics
│   └── weights/
│       └── best_model.pth       # Trained U-Net weights (best validation loss)
│
├── inference/
│   ├── scripts/
│   │   ├── inference.py         # Local inference script (Windows/Linux/Mac)
│   │   └── inference_colab.ipynb # Colab notebook version
│   └── outputs/                 # Sample annotated output images
│
├── measurement/
│   ├── scripts/
│   │   └── measure.py           # Pixel-to-mm measurement pipeline
│   └── results/
│       ├── measurement_report.csv   # Per-image accuracy table
│       └── error_analysis.png       # MAE and MPE bar charts
│
├── docs/
│   ├── CALIBRATION_REPORT.md
│   ├── DATASET_CARD.md
│   ├── TRAINING_REPORT.md
│   ├── MEASUREMENT_REPORT.md
│   └── SETUP.md
│
├── Calculator_Segmentation.ipynb  # Full training notebook (Colab)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/noman9596/Image-segmentation-and-detect-measurements.git
cd Image-segmentation-and-detect-measurements
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### 2. Run inference on a single image

```bash
python inference/scripts/inference.py --image path/to/your/image.jpg
```

### 3. Run on a folder of images

```bash
python inference/scripts/inference.py --folder path/to/folder/
```

When the script runs, it displays the undistorted image with a coordinate
grid. You read the pixel coordinates of the ruler's start and end marks
from the grid, type them into the terminal, and the system computes the
measurement automatically from there.

Full installation and usage details are in `docs/SETUP.md`.

---

## Pipeline Overview

The pipeline follows the measurement workflow specified in the assessment,
applied to a Casio fx-991ES PLUS scientific calculator as the target object.

**Stage 1 — Camera Calibration**

A 7x9 inner-corner checkerboard with 20mm squares was printed on A4 paper
and photographed from 22 different angles and distances using a smartphone
camera. OpenCV's `calibrateCamera` function was used to compute the intrinsic
matrix and distortion coefficients. Every image used in the rest of the
pipeline is passed through `cv2.undistort()` before any further processing.
Without this step, pixel distances near the edges of the frame do not map
linearly to real-world distances, which would make measurements unreliable.

**Stage 2 — Data Collection and Labelling**

96 images of the calculator were captured using the same camera used for
calibration. The images cover a variety of backgrounds (plain surfaces,
fabric, paper, outdoor concrete), lighting conditions (indoor, natural
daylight, direct sunlight), and orientations (portrait, landscape, slight
angles). Each image was labelled using Roboflow with a polygon segmentation
mask drawn tightly around the calculator boundary. Labels were exported in
COCO JSON format and split into 80 training, 21 validation, and 11 test
images.

**Stage 3 — Model Training**

A U-Net with a ResNet-34 ImageNet-pretrained encoder was used for segmentation due to its encoder decoder
 structure and skip connections, which preserve boundary details important for 
 accurate measurements. The model was trained for 50 epochs on Google Colab (T4 GPU) 
 using a BCE + Dice loss, Adam optimizer, and a learning rate of 1e-4. As each image 
 contains only one calculator, semantic segmentation effectively serves as instance segmentation,
  making U-Net a simpler and suitable alternative to Mask R-CNN.

**Stage 4 — Pixel-to-MM Measurement**

A 150 mm ruler is included in each image for calibration. The 0 mm and 150 mm points are
 marked to compute a pixels-per-millimetre ratio. The U-Net segments the calculator, and 
 cv2.minAreaRect extracts its rotated bounding box. The box dimensions are converted from
  pixels to millimetres using the calibration ratio and compared with ground-truth ruler
   measurements.

---

## Results

**Segmentation performance on the test set:**

| Metric | Score |
|---|---|
| IoU | 0.8728 |
| Precision | 0.9561 |
| Recall | 0.8998 |
| F1 Score | 0.9271 |

These results are strong for a dataset of 80 training images and demonstrate
that the transfer learning approach from ImageNet generalises well to this
single-class detection task.

**Measurement accuracy across 5 test instances:**

| Metric | Value |
|---|---|
| MAE Width | 10.60 mm |
| MAE Height | 18.64 mm |
| MPE Width | 13.76% |
| MPE Height | 11.54% |

The best result achieved 0.83 mm width error (1.1%) and 3.08 mm height error (1.9%), 
demonstrating good accuracy for a single-camera, non-contact measurement system. Higher
 average errors were mainly caused by outliers due to inaccurate ruler calibration or larger
  viewing angles than those seen during training.

---

## Design Decisions

**Why U-Net instead of YOLO or Mask R-CNN**

YOLO and Roboflow models were excluded per the assessment requirements.
Mask R-CNN was considered but presents significant setup complexity on
Windows without a dedicated GPU. U-Net with a ResNet-34 backbone achieves
comparable mask quality for single-instance segmentation, trains faster
on limited data, and is simpler to deploy. The architecture was originally
designed for small medical datasets, making it well-suited to the 80-image
training set used here.

**Why manual ruler reference instead of automatic detection**
Automatic ruler detection using Hough transforms and contour-based methods was tested
 but proved unreliable, failing in many cases due to background similarities. A manual 
 two-point calibration approach was therefore used, providing consistent and error-free
  reference measurements. In practical applications, this could be automated using ArUco
   markers for robust, sub-pixel accurate detection.

**Why the same camera must be used throughout**

Camera calibration computes parameters specific to the optical system that
captured the calibration images. If a different camera is used to capture
measurement images, the distortion model will not apply correctly and
pixel-to-mm conversions will be wrong even if the images look similar.

---

## Limitations

The 2.3-pixel calibration error exceeded the desired threshold due to slight bending 
of the printed checkerboard; a rigid calibration target would improve accuracy.
 Although the 80-image dataset achieved 0.87 IoU, its limited size reduced robustness 
 to unseen conditions, with blur and steep viewing angles causing larger errors.
  Additionally, the system currently requires a visible ruler for calibration,
   which could be replaced by a fixed camera setup or a calibrated reference marker
    for practical deployment.

---

## Possible Improvements

Replacing the paper checkerboard with a rigid printed or laser-cut
calibration board would bring the reprojection error below 0.3 pixels and
improve the reliability of undistortion.

Expanding the dataset to 300 or more images with greater variation in
background, lighting, and camera distance would improve both segmentation
accuracy and measurement consistency.

Replacing the manual ruler reference with an ArUco marker of known physical
size would make the full pipeline automatic and remove the need for operator
input during inference.

A fixed camera setup with a known sensor size and focal length would allow
measurements to be derived from camera intrinsics alone, removing the
dependency on a reference object in every image.

---

## Documentation

All technical documentation is in the `docs/` folder:

- `CALIBRATION_REPORT.md` — calibration method, intrinsic parameters, reprojection error
- `DATASET_CARD.md` — object description, collection strategy, labelling process, statistics
- `TRAINING_REPORT.md` — model architecture, training configuration, metrics, loss curves
- `MEASUREMENT_REPORT.md` — pixel-to-mm derivation, accuracy table, error analysis
- `SETUP.md` — full installation and usage instructions

---

## Dependencies

Python 3.8 or higher is required. All dependencies are listed in
`requirements.txt` and can be installed with:

```bash
pip install -r requirements.txt
```

The model was trained on Google Colab using a free T4 GPU. Inference runs
on CPU without any GPU requirement.

---

## Author

Noman Shahid
Computer Science Graduate, FAST National University of Computer and Emerging Sciences
GitHub: https://github.com/noman9596/Image-segmentation-and-detect-measurements
