

import argparse
import os
import sys
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ── Check dependencies ────────────────────────────────────────
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("ERROR: segmentation_models_pytorch not installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# ── CONFIGURATION — edit these paths if needed ────────────────
CAMERA_MATRIX_PATH = "calibration/camera_matrix.npy"
DIST_COEFFS_PATH   = "calibration/dist_coeffs.npy"
MODEL_PATH         = "models/best_model.pth"
OUTPUT_DIR         = "outputs"

REFERENCE_MM  = 150.0   
GT_WIDTH_MM   = 77.0    
GT_HEIGHT_MM  = 161.5   
INPUT_SIZE    = 512   
DISPLAY_SCALE = 0.20    


os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_calibration():
    if not os.path.exists(CAMERA_MATRIX_PATH):
        print(f"ERROR: Camera matrix not found at '{CAMERA_MATRIX_PATH}'")
        print("Place camera_matrix.npy inside the calibration/ folder.")
        sys.exit(1)
    if not os.path.exists(DIST_COEFFS_PATH):
        print(f"ERROR: Distortion coefficients not found at '{DIST_COEFFS_PATH}'")
        sys.exit(1)
    mtx  = np.load(CAMERA_MATRIX_PATH)
    dist = np.load(DIST_COEFFS_PATH)
    print(f"Calibration loaded")
    print(f"   fx={mtx[0,0]:.1f}  fy={mtx[1,1]:.1f}  "
          f"cx={mtx[0,2]:.1f}  cy={mtx[1,2]:.1f}")
    return mtx, dist


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'")
        print("Place best_model.pth inside the models/ folder.")
        sys.exit(1)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict):
        try:
            model.load_state_dict(state)
        except Exception:
            try:
                model = torch.load(MODEL_PATH, map_location=device)
            except Exception as e:
                print(f"ERROR loading model: {e}")
                sys.exit(1)
    else:
        model = state

    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newmtx)
    x, y, rw, rh = roi
    if rw > 10 and rh > 10:
        dst = dst[y:y+rh, x:x+rw]
    return dst


def get_mask(img_bgr, model, device):
    ih, iw = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    rgb  = (rgb - mean) / std
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    if isinstance(out, (list, tuple)):
        out = out[0]
    mask = out.squeeze().cpu().numpy()
    if mask.min() < -0.5 or mask.max() > 1.5:
        mask = 1.0 / (1.0 + np.exp(-mask))
    mask = (mask > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)


def measure_from_mask(mask, px_per_mm):
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m,    cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return None
    rect   = cv2.minAreaRect(c)
    d1, d2 = rect[1]
    h_px   = max(d1, d2)
    w_px   = min(d1, d2)
    return {
        'w_mm'    : round(w_px / px_per_mm, 2),
        'h_mm'    : round(h_px / px_per_mm, 2),
        'w_px'    : round(w_px, 1),
        'h_px'    : round(h_px, 1),
        'rect'    : rect,
        'contour' : c,
    }


def build_annotated(undist, mask, meas, pt1, pt2, px_per_mm):
    ann = undist.copy()
    ov  = ann.copy()
    ov[mask > 0] = [0, 200, 0]
    ann = cv2.addWeighted(ann, 0.70, ov, 0.30, 0)

    # Rotated bounding box
    box = cv2.boxPoints(meas['rect']).astype(int)
    cv2.drawContours(ann, [box], 0, (0, 220, 0), 3)
    cv2.drawContours(ann, [meas['contour']], -1, (0, 255, 100), 2)

    # Ruler reference line
    cv2.line(ann, pt1, pt2, (0, 130, 255), 4)
    cv2.circle(ann, pt1, 16, (0, 80, 255), -1)
    cv2.circle(ann, pt2, 16, (0, 80, 255), -1)
    cv2.putText(ann, "0 mm",  (pt1[0]-10, pt1[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
    cv2.putText(ann, f"{REFERENCE_MM:.0f} mm", (pt2[0]-20, pt2[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)

    # Info box
    cv2.rectangle(ann, (8, 8), (620, 260), (0, 0, 0), -1)
    cv2.rectangle(ann, (8, 8), (620, 260), (0, 220, 0), 2)
    lines = [
        (f"Width  : {meas['w_mm']} mm   (GT = {GT_WIDTH_MM} mm)", (0, 255,   0)),
        (f"Height : {meas['h_mm']} mm   (GT = {GT_HEIGHT_MM} mm)", (0, 255,   0)),
        (f"px/mm  : {px_per_mm:.4f}",                              (0, 180, 255)),
        (f"Ref    : {REFERENCE_MM:.0f} mm = {pt1}→{pt2}",         (0, 180, 255)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(ann, txt, (18, 58 + i * 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, col, 2)
    return ann


def show_grid_image(undist, display_scale, fname):
    """Display image with coordinate grid so user can read ruler coords."""
    h, w = undist.shape[:2]
    dw   = int(w * display_scale)
    dh   = int(h * display_scale)
    disp = cv2.resize(undist, (dw, dh))

    
    for gx in range(0, dw, 50):
        cv2.line(disp, (gx, 0), (gx, dh), (0, 200, 200), 1)
        cv2.putText(disp, str(gx), (gx+2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)
    for gy in range(0, dh, 50):
        cv2.line(disp, (0, gy), (dw, gy), (0, 200, 200), 1)
        cv2.putText(disp, str(gy), (2, gy+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

 
    instructions = [
        "FIND YOUR RULER in the image below",
        f"Note coords at ruler START (0mm) and END ({REFERENCE_MM:.0f}mm)",
        "Grid lines labeled every 50px — read directly from grid",
        "Press any key to close this window",
    ]
    for i, txt in enumerate(instructions):
        cv2.putText(disp, txt, (5, dh - 80 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3)
        cv2.putText(disp, txt, (5, dh - 80 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1)

    cv2.putText(disp, f"FILE: {fname}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 2)
    cv2.putText(disp, f"Display: {dw}x{dh}  |  Full: {w}x{h}  |  scale=x{1/display_scale:.0f}",
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (200, 200, 200), 1)

    cv2.imshow(f"GRID VIEW — {fname} (press any key to close)", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dw, dh


def get_ruler_coords_from_user(dw, dh, display_scale, fname):
    """Ask user to type in ruler coordinates."""
    scale = 1.0 / display_scale
    print()
    print("─" * 55)
    print(f"  ENTER RULER COORDINATES for: {fname}")
    print("─" * 55)
    print(f"  Display image was: {dw} x {dh} pixels")
    print(f"  Coordinates are in DISPLAY pixels (from the grid)")
    print(f"  The script auto-converts them to full resolution")
    print()
    print(f"  Point 1 = ruler 0 mm mark")
    print(f"  Point 2 = ruler {REFERENCE_MM:.0f} mm mark")
    print()

    while True:
        try:
            raw = input(f"  Enter P1 x,y (ruler start):  ").strip()
            p1x, p1y = [int(v.strip()) for v in raw.split(',')]
            raw = input(f"  Enter P2 x,y (ruler end):    ").strip()
            p2x, p2y = [int(v.strip()) for v in raw.split(',')]

            # Validate inside display bounds
            if not (0 <= p1x <= dw and 0 <= p1y <= dh):
                print(f"  P1 ({p1x},{p1y}) is outside display ({dw}x{dh}) — try again")
                continue
            if not (0 <= p2x <= dw and 0 <= p2y <= dh):
                print(f"   P2 ({p2x},{p2y}) is outside display ({dw}x{dh}) — try again")
                continue

            # Convert to full resolution
            pt1 = (int(p1x * scale), int(p1y * scale))
            pt2 = (int(p2x * scale), int(p2y * scale))

            # Ruler length sanity check
            ruler_px = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
            if ruler_px < 30:
                print(f" Points too close ({ruler_px:.0f}px) — are they correct?")
                retry = input("  Retry? (y/n): ").strip().lower()
                if retry == 'y':
                    continue

            print(f"\n   Full-res coords: P1={pt1}  P2={pt2}")
            print(f"  Ruler span     : {ruler_px:.1f} px")
            print(f"   px/mm ratio    : {ruler_px/REFERENCE_MM:.4f}")
            return pt1, pt2, ruler_px

        except (ValueError, Exception) as e:
            print(f" Invalid input ({e}) — enter as: x,y   e.g.  120,350")


def process_single_image(img_path, mtx, dist, model, device,
                          display_scale, batch_coords=None):
    """
    Process one image end-to-end.
    batch_coords: (p1x,p1y,p2x,p2y) in display pixels — if None, asks user
    """
    fname    = Path(img_path).name
    print(f"\n{'='*55}")
    print(f"  Processing: {fname}")
    print(f"{'='*55}")

    # 1. Load
    img = cv2.imread(str(img_path))
    if img is None:
        print(f" Cannot read image: {img_path}")
        return None

    # 2. Undistort
    undist = undistort_image(img, mtx, dist)
    h_u, w_u = undist.shape[:2]
    print(f"  Image size after undistortion: {w_u} x {h_u}")

    # 3. Show grid + get ruler coords
    if batch_coords is not None:
        # Batch mode — coords already provided
        p1x, p1y, p2x, p2y = batch_coords
        scale = 1.0 / display_scale
        dw = int(w_u * display_scale)
        dh = int(h_u * display_scale)
        pt1 = (int(p1x * scale), int(p1y * scale))
        pt2 = (int(p2x * scale), int(p2y * scale))
        ruler_px = np.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)
        print(f"  Batch mode coords: pt1={pt1} pt2={pt2}")
    else:
        # Interactive mode — show grid window, user types coords
        dw, dh = show_grid_image(undist, display_scale, fname)
        pt1, pt2, ruler_px = get_ruler_coords_from_user(dw, dh, display_scale, fname)

    px_per_mm = ruler_px / REFERENCE_MM

    # 4. Segment
    print(f"\n Running U-Net segmentation...")
    mask     = get_mask(undist, model, device)
    coverage = (mask > 0).sum() / mask.size * 100
    print(f"  Mask coverage: {coverage:.1f}%")

    if coverage < 0.3:
        print(" Very small mask — model may not have detected the calculator")

    # 5. Measure
    meas = measure_from_mask(mask, px_per_mm)
    if meas is None:
        print("   Could not find calculator contour in mask")
        return None

    w_mm  = meas['w_mm']
    h_mm  = meas['h_mm']
    err_w = abs(w_mm - GT_WIDTH_MM)
    err_h = abs(h_mm - GT_HEIGHT_MM)
    pct_w = err_w / GT_WIDTH_MM  * 100
    pct_h = err_h / GT_HEIGHT_MM * 100

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  MEASUREMENT RESULTS                    │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Width   : {w_mm:>8.2f} mm  (GT={GT_WIDTH_MM} mm)  │")
    print(f"  │  Height  : {h_mm:>8.2f} mm  (GT={GT_HEIGHT_MM} mm) │")
    print(f"  │  Err W   : {err_w:>8.2f} mm  ({pct_w:.1f}%)          │")
    print(f"  │  Err H   : {err_h:>8.2f} mm  ({pct_h:.1f}%)          │")
    print(f"  │  px/mm   : {px_per_mm:>8.4f}                    │")
    print(f"  └─────────────────────────────────────────┘")

    # 6. Annotate + save
    ann      = build_annotated(undist, mask, meas, pt1, pt2, px_per_mm)
    out_path = os.path.join(OUTPUT_DIR, f"result_{fname}")
    cv2.imwrite(out_path, ann)
    print(f"\n   Saved: {out_path}")

    # 7. Show result window
    preview_h = 700
    preview_w = int(ann.shape[1] * preview_h / ann.shape[0])

    # Side by side: original | mask | result
    orig_r  = cv2.resize(undist, (preview_w, preview_h))
    mask_r  = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                          (preview_w, preview_h))
    ann_r   = cv2.resize(ann, (preview_w, preview_h))

    # Labels
    for panel, label in [(orig_r,"1. Undistorted"),
                          (mask_r,"2. Mask"),
                          (ann_r, "3. Result")]:
        cv2.putText(panel, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    combined = np.hstack([orig_r, mask_r, ann_r])
    cv2.imshow(f"Result — {fname}  (press any key)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        'image'        : fname,
        'width_mm'     : w_mm,
        'height_mm'    : h_mm,
        'gt_width_mm'  : GT_WIDTH_MM,
        'gt_height_mm' : GT_HEIGHT_MM,
        'err_w_mm'     : round(err_w, 2),
        'err_h_mm'     : round(err_h, 2),
        'pct_err_w'    : round(pct_w, 2),
        'pct_err_h'    : round(pct_h, 2),
        'px_per_mm'    : round(px_per_mm, 4),
        'ruler_px'     : round(ruler_px, 1),
        'mask_coverage': round(coverage, 2),
        'output_path'  : out_path,
    }


def save_report(results):
    """Save CSV + JSON accuracy report."""
    if not results:
        return

    import csv

    csv_path  = os.path.join(OUTPUT_DIR, "measurement_report.csv")
    json_path = os.path.join(OUTPUT_DIR, "measurement_report.json")

    # CSV
    keys = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

   
    err_w_list = [r['err_w_mm'] for r in results]
    err_h_list = [r['err_h_mm'] for r in results]
    pct_w_list = [r['pct_err_w'] for r in results]
    pct_h_list = [r['pct_err_h'] for r in results]

    summary = {
        'timestamp'        : datetime.now().isoformat(),
        'images_processed' : len(results),
        'gt_width_mm'      : GT_WIDTH_MM,
        'gt_height_mm'     : GT_HEIGHT_MM,
        'reference_mm'     : REFERENCE_MM,
        'mae_width_mm'     : round(np.mean(err_w_list), 2),
        'mae_height_mm'    : round(np.mean(err_h_list), 2),
        'mpe_width_pct'    : round(np.mean(pct_w_list), 2),
        'mpe_height_pct'   : round(np.mean(pct_h_list), 2),
        'results'          : results,
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*55}")
    print(f"  Images processed : {len(results)}")
    print(f"  MAE Width        : {summary['mae_width_mm']:.2f} mm")
    print(f"  MAE Height       : {summary['mae_height_mm']:.2f} mm")
    print(f"  MPE Width        : {summary['mpe_width_pct']:.2f} %")
    print(f"  MPE Height       : {summary['mpe_height_pct']:.2f} %")
    print(f"{'='*55}")
    print(f"  CSV  saved: {csv_path}")
    print(f"  JSON saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculator Measurement Pipeline — Segment & Measure in mm",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
EXAMPLES:
  Single image (interactive):
    python inference.py --image test_images/calc.jpg

  Folder of images (interactive one by one):
    python inference.py --folder test_images/

  Single image with custom display scale:
    python inference.py --image calc.jpg --scale 0.15

  Batch mode (no windows — provide coords in coords.json):
    python inference.py --folder test_images/ --batch coords.json

COORDS JSON FORMAT (for batch mode):
  {
    "image1.jpg": [120, 350, 420, 352],
    "image2.jpg": [80,  300, 380, 302]
  }
  Values are display pixel coords: [P1x, P1y, P2x, P2y]
        """
    )
    parser.add_argument('--image',  type=str, help='Path to a single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--scale',  type=float, default=DISPLAY_SCALE,
                        help=f'Display scale for grid (default={DISPLAY_SCALE})')
    parser.add_argument('--batch',  type=str, default=None,
                        help='JSON file with pre-filled coords (batch mode)')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        sys.exit(1)

    display_scale = args.scale

    # ── Load calibration + model ──────────────────────────────
    print("\n" + "="*55)
    print("  CALCULATOR MEASUREMENT PIPELINE")
    print("="*55)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device : {device}")

    mtx, dist = load_calibration()
    model     = load_model(device)

    # ── Load batch coords if provided ────────────────────────
    batch_coords_map = {}
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"ERROR: batch coords file not found: {args.batch}")
            sys.exit(1)
        with open(args.batch) as f:
            batch_coords_map = json.load(f)
        print(f" Batch mode: loaded coords for {len(batch_coords_map)} images")

    # ── Collect image paths ───────────────────────────────────
    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        image_paths = [args.image]
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"ERROR: Folder not found: {args.folder}")
            sys.exit(1)
        for ext in ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG']:
            image_paths.extend(Path(args.folder).glob(ext))
        image_paths = sorted(image_paths)
        if not image_paths:
            print(f"ERROR: No images found in {args.folder}")
            sys.exit(1)
        print(f"\n Found {len(image_paths)} images in folder")

    # ── Process each image ────────────────────────────────────
    all_results = []
    for img_path in image_paths:
        fname  = Path(img_path).name
        bcoords = batch_coords_map.get(fname, None)

        result = process_single_image(
            img_path, mtx, dist, model, device,
            display_scale, batch_coords=bcoords
        )
        if result:
            all_results.append(result)

        if len(image_paths) > 1:
            cont = input("\n  Process next image? (y/n): ").strip().lower()
            if cont != 'y':
                break

    # ── Save report ───────────────────────────────────────────
    if all_results:
        save_report(all_results)
    else:
        print("\n  No successful results to report.")

    print("\n Done. Check the 'outputs/' folder for results.\n")


if __name__ == '__main__':
    main()