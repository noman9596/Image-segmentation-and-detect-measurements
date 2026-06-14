# Setup & Installation Guide

Complete step-by-step instructions to reproduce the full pipeline —
from environment setup to running inference and measurement.

---

## System Requirements

| Component | Requirement |
|---|---|
| OS | Windows 10/11, Ubuntu 20.04+, or macOS |
| Python | 3.8 or higher |
| RAM | Minimum 8 GB |
| GPU | Optional (NVIDIA CUDA) — CPU works fine for inference |
| Storage | ~2 GB free space |
| Camera | Same device used during calibration (for new captures) |

---

## Repository Structure

```
Image-segmentation-and-detect-measurements/
│
├── calibration/
│   ├── images/                  # Checkerboard calibration images
│   ├── scripts/
│   │   └── calibrate.py         # Camera calibration script
│   ├── camera_matrix.npy        # Saved intrinsic matrix
│   ├── dist_coeffs.npy          # Saved distortion coefficients
│   └── calibration_params.json  # Human-readable calibration params
│
├── dataset/
│   ├── train/                   # 70 training images + COCO JSON
│   ├── valid/                   # 15 validation images + COCO JSON
│   └── test/                    # 11 test images + COCO JSON
│
├── models/
│   ├── best_model.pth           # Trained U-Net weights
│   └── model_info.json          # Architecture and metrics summary
│
├── inference/
│   └── inference.py             # Main inference script
│
├── measurement/
│   ├── outputs/                 # Annotated result images
│   └── measurement_report.csv   # Accuracy table
│
├── docs/
│   ├── CALIBRATION_REPORT.md
│   ├── DATASET_CARD.md
│   ├── TRAINING_REPORT.md
│   ├── MEASUREMENT_REPORT.md
│   └── SETUP.md                 # This file
│
├── requirements.txt
└── README.md
```

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/noman9596/Image-segmentation-and-detect-measurements.git
cd Image-segmentation-and-detect-measurements
```

---

## Step 2 — Create Python Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages. Expected time: 3–5 minutes.

**Contents of requirements.txt:**
```
torch==2.0.1
torchvision==0.15.2
segmentation-models-pytorch==0.3.3
opencv-python==4.8.0.76
numpy==1.24.3
albumentations==1.3.1
matplotlib==3.7.2
Pillow==10.0.0
pandas==2.0.3
```

> **GPU users:** Replace the torch line with the CUDA version:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Step 4 — Verify Files Are in Place

Before running anything, confirm these files exist:

```bash
# Check calibration files
ls calibration/camera_matrix.npy
ls calibration/dist_coeffs.npy

# Check model weights
ls models/best_model.pth
```

If any file is missing, see the sections below.

---

## Step 5 — Run Camera Calibration (Skip if Already Done)

Only needed if you want to recalibrate with your own camera.

```bash
# Place your checkerboard images in calibration/images/
python calibration/scripts/calibrate.py
```

This generates:
- `calibration/camera_matrix.npy`
- `calibration/dist_coeffs.npy`
- `calibration/calibration_params.json`
- `calibration/undistortion_comparison.png`

**Target reprojection error:** below 1.0 px (below 0.5 px is excellent)

---

## Step 6 — Run Inference (Measurement Pipeline)

### Single Image
```bash
python inference/inference.py --image path/to/your/image.jpg
```

### Folder of Images
```bash
python inference/inference.py --folder path/to/your/folder/
```

### Custom Display Scale (if image appears too large or small)
```bash
python inference/inference.py --image image.jpg --scale 0.15
```

---

## How the Inference Works — User Workflow

When you run inference on an image, the pipeline follows these steps:

### 1. Image Loads and Undistorts
The image is automatically undistorted using the saved calibration
parameters (`camera_matrix.npy` + `dist_coeffs.npy`).

### 2. Grid Window Opens
A window opens showing the undistorted image with a cyan coordinate
grid overlaid every 50 pixels. This grid lets you read pixel positions
directly from the image.

```
┌─────────────────────────────────────┐
│  0    50   100  150  200  250 ...   │
│0 ┼────┼────┼────┼────┼────┼         │
│  │    │    │    │    │    │         │
│50┼────┼────┼────┼────┼────┼         │
│  │  [your image here]              │
│  │                                 │
│  └── Read coords where ruler is ──┘│
└─────────────────────────────────────┘
```

**Press any key** to close the grid window.

### 3. You Enter Ruler Coordinates
After closing the grid window, the terminal asks:

```
Enter P1 x,y (ruler start — 0mm mark):   120,350
Enter P2 x,y (ruler end   — 150mm mark):  420,352
```

Type the x,y coordinates you read from the grid for:
- **P1** = where your ruler starts (0mm mark)
- **P2** = where your ruler ends (150mm mark = 15cm)

Coordinates are in **display pixels** — the script automatically
converts them to full resolution.

### 4. Results Display
A result window opens showing three panels:

| Panel | Content |
|---|---|
| 1. Undistorted Input | Original image after lens correction |
| 2. U-Net Mask | Binary segmentation mask from model |
| 3. Measurement Result | Annotated image with mm measurements |

Terminal also prints:
```
┌─────────────────────────────────────────┐
│  MEASUREMENT RESULTS                    │
├─────────────────────────────────────────┤
│  Width   :    73.74 mm  (GT=77.0 mm)   │
│  Height  :   145.06 mm  (GT=161.5 mm)  │
│  Err W   :     3.26 mm  (4.2%)         │
│  Err H   :    16.44 mm  (10.2%)        │
│  px/mm   :   19.0740                   │
└─────────────────────────────────────────┘
```

### 5. Output Saved
Results are saved automatically to `outputs/`:
- `result_<imagename>.jpg` — annotated image
- `measurement_report.csv` — accuracy table (after all images)
- `measurement_report.json` — full summary with MAE/MPE

---

## Running on Google Colab (Training Only)

Training was performed on Google Colab using a free T4 GPU.
To retrain the model:

1. Upload the `dataset/` folder to Google Drive
2. Open the training notebook: `models/train_unet.ipynb`
3. Set `DATASET_PATH` to your Google Drive path
4. Runtime → Change runtime type → **T4 GPU**
5. Run all cells

Training takes approximately 20–30 minutes on T4 GPU for 50 epochs.

---

## Troubleshooting

### "No module named segmentation_models_pytorch"
```bash
pip install segmentation-models-pytorch
```

### "Camera matrix not found"
Make sure `camera_matrix.npy` is in the `calibration/` folder.
Run `python calibration/scripts/calibrate.py` to regenerate it.

### "Model not found"
Make sure `best_model.pth` is in the `models/` folder.
Download it from the Google Drive link in the README or retrain.

### "Cannot read image"
Check the file path is correct and the image is not corrupted:
```bash
python -c "import cv2; img=cv2.imread('your_image.jpg'); print(img.shape)"
```

### Grid window does not open (headless server)
Use batch mode with a coords JSON file:
```bash
python inference/inference.py --folder images/ --batch coords.json
```

### Mask is empty (all black)
The model did not detect the calculator. Check that:
- The calculator is clearly visible and in focus
- The image is not too dark or overexposed
- You are using the same camera that was used during calibration

### Measurements are very inaccurate
Common causes:
- Wrong ruler coordinates entered (P1/P2 swapped or misread)
- Ruler not flat in the image (tilted significantly toward camera)
- Image taken with a different camera than the one calibrated


---

## Contact

For issues or questions about this project, refer to the repository:
https://github.com/noman9596/Image-segmentation-and-detect-measurements



