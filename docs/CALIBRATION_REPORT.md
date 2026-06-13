# Camera Calibration

## Objective

Camera calibration was performed to estimate the camera's intrinsic parameters and lens distortion coefficients. These parameters are required to correct image distortion and improve the accuracy of computer vision applications.



## Calibration Setup

- Checkerboard Size: 7 × 9 inner corners
- Square Size: 20 mm
- Corner Detection Method: OpenCV `findChessboardCornersSB()`
- Calibration Function: OpenCV `calibrateCamera()`



## Methodology

1. Collected multiple checkerboard images from different viewpoints.
2. Detected checkerboard corners in each image.
3. Performed initial camera calibration.
4. Calculated reprojection error for every image.
5. Removed images with reprojection error greater than **1.5 px**.
6. Recalibrated using only high-quality images.
7. Generated undistorted images for visual verification.



## Results

| Metric | Value |
|----------|----------|
| Total Images | 22 |
| Valid Images | 22 |
| Images Used After Filtering | 17 |
| Final Reprojection Error | 0.549 px |
| Calibration Rating | Good |

A reprojection error of **0.549 px** indicates that the camera parameters accurately model the camera and can be used for distortion correction.



## Conclusion

The camera was successfully calibrated using checkerboard images. Low-quality images were removed through reprojection error analysis, and recalibration was performed on the filtered dataset. The final calibration achieved a reprojection error of **0.549 px**, providing reliable distortion correction for further computer vision tasks.