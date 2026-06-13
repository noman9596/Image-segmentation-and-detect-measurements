# Reference Scale Measurement

## Design Decision

The assessment recommends automatic reference object detection. Initially, automatic ruler detection using edge detection, contour extraction, and Hough Transform techniques was investigated.

However, the ruler used in this project had low visibility and weak edge features, resulting in inconsistent and unreliable detections. Factors affecting performance included:

* Low contrast between the ruler and background
* Partial visibility of ruler markings
* Lighting variations across images

To ensure accurate measurements, a manual calibration approach was adopted.

## Method

The user manually selects two known points on the ruler:

* 0 mm mark
* 150 mm mark

The Euclidean distance between these points is calculated and used to determine the pixel-to-millimetre scale:

```text
pixels_per_mm = distance(pt1, pt2) / 150
```

This scale factor is then used to convert pixel measurements into real-world dimensions.

## Justification

The manual approach was chosen because:

* It provides higher measurement accuracy than unreliable automatic detection.
* It eliminates detection errors caused by poor ruler visibility.
* It produces consistent and reproducible measurements.
* It is a common practice in calibrated measurement systems where reference points are verified by the operator.

## Future Improvement

Future versions can replace manual point selection with automatic reference detection using dedicated fiducial markers such as ArUco markers, AprilTags, or high-contrast reference objects.
