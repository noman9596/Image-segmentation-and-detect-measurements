# Image Segmentation and Metric Measurement Pipeline

End-to-end computer vision pipeline that segments a physical object and measures
its real-world dimensions (width & height in mm) using camera calibration.

## Pipeline Overview
1. Camera calibration (OpenCV checkerboard)
2. Dataset collection and segmentation labelling
3. Mask R-CNN model training (Detectron2)
4. Pixel-to-mm measurement using calibrated reference object

## Quick Start
See [docs/SETUP.md](docs/SETUP.md) for full installation and run instructions.
