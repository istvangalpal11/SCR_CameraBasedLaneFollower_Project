# Metric BEV Calibration for the BFMC Car

## Goal

The BEV should not only make lane lines look parallel. It should represent the road plane with a consistent metric scale:

- one x-pixel always represents the same lateral distance;
- one y-pixel always represents the same forward distance.

This keeps the existing BFMC-style pipeline valid with `xm_per_pix`, `ym_per_pix`, sliding windows, polynomial fitting, curvature calculations, and MPC reference geometry.

## Why the current hand-tuned trapezoid is not enough

A manually selected trapezoid can make straight lane boundaries look parallel, but that does not guarantee that equal real-world distances map to equal pixel distances everywhere in the BEV.

Bad example:

- 10 cm near the car -> 25 pixels
- 10 cm in the middle -> 18 pixels
- 10 cm farther away -> 10 pixels

Then one constant `ym_per_pix` is not valid.

The target is that the same real distance gives approximately the same pixel distance everywhere in the useful BEV area.

---

# Step-by-step procedure when the BFMC car is available

## 1. Fix the camera in its final position

Mount the camera exactly as it will be used on the car.

Do not change afterwards:

- camera height
- pitch
- yaw
- roll
- lens/focus
- image resolution

If the camera position changes, repeat or at least revalidate the ground-plane calibration.

## 2. Perform intrinsic chessboard camera calibration

Use a printed chessboard with a precisely known square size.

Take multiple pictures:

- close and farther away
- left and right parts of the image
- different orientations
- different positions

Use OpenCV to estimate:

- camera matrix `K`
- distortion coefficients `dist`

Typical functions:

- `cv2.findChessboardCorners`
- `cv2.cornerSubPix`
- `cv2.calibrateCamera`

Save `K` and `dist`.

## 3. Undistort before BEV calibration

All BEV calibration points should be taken from the undistorted image.

Runtime should later begin with:

    camera image
        -> undistort
        -> BEV transform

## 4. Put known calibration points on the flat road plane

Place a chessboard, calibration grid, or accurately measured markers on the same flat ground plane on which the car drives.

Define vehicle coordinates:

- `X` = lateral position
- `Y` = forward distance

Use points that cover:

- near field
- middle field
- farther MPC look-ahead region

Do not rely on only four points if more reliable points are available.

## 5. Record the corresponding image pixels

For every known ground point `(X, Y)`, determine its pixel coordinate `(u, v)` in the undistorted camera image.

You now have correspondences:

    camera pixel (u, v)
            <->
    real ground point (X, Y)

## 6. Choose the physical BEV area

Decide what road area the controller really needs.

Example only:

    X_min = -0.5 m
    X_max = +0.5 m

    Y_min = 0.10 m
    Y_max = 2.0 m

Choose the final values later from the real BFMC camera and MPC horizon.

## 7. Choose `xm_per_pix` and `ym_per_pix`

Keep the existing BFMC idea.

Choose:

    xm_per_pix = ... m/pixel
    ym_per_pix = ... m/pixel

They do not need to be equal.

Then:

    BEV_width_px =
        (X_max - X_min) / xm_per_pix

    BEV_height_px =
        (Y_max - Y_min) / ym_per_pix

Now every BEV pixel has a known physical meaning.

## 8. Convert real-world points to BEV pixels

For each real point `(X, Y)`, calculate where it should appear in the BEV.

Conceptually:

    bev_x = (X - X_min) / xm_per_pix

If the car is at the bottom of the BEV:

    bev_y = (Y_max - Y) / ym_per_pix

Use one coordinate convention consistently.

## 9. Estimate the homography

Calculate the projective transform between:

    undistorted camera pixels
            ->
    metric BEV pixels

Prefer using multiple calibration points with:

    cv2.findHomography(...)

rather than manually choosing an arbitrary visual trapezoid.

Save the resulting homography matrix `H`.

Runtime then uses:

    cv2.warpPerspective(image, H, BEV_SIZE)

---

# Validation

Do not accept the calibration only because lane lines look parallel.

## 10. Check known longitudinal distances

Place points with equal real forward spacing.

Example:

    A -> B = 10 cm
    B -> C = 10 cm
    C -> D = 10 cm

Measure them in the BEV.

They should satisfy approximately:

    A-B ~= B-C ~= C-D

from the bottom to the top of the useful BEV region.

If the same 10 cm becomes smaller toward the top, the BEV is still not metrically correct.

## 11. Check known lateral distances

Do the same in the x direction.

A known 20 cm lateral distance should represent approximately the same number of BEV pixels near, middle, and farther away.

## 12. Check chessboard square geometry

If every chessboard square has the same physical size, then in a correct BEV they should also have approximately the same pixel dimensions throughout the useful region.

Far squares may look blurrier because fewer original camera pixels represent them, but their metric size should not systematically shrink.

## 13. Check lane width

On a straight section, measure lane width at several y positions.

Compute:

    lane_width_m =
        lane_width_pixels * xm_per_pix

It should remain approximately constant.

---

# Runtime pipeline afterwards

The planned pipeline can remain very close to BFMC:

    camera frame
        ->
    undistort
        ->
    metric BEV using saved H
        ->
    Top-Hat / threshold / morphology
        ->
    sliding-window lane search
        ->
    polynomial fitting
        ->
    convert using xm_per_pix and ym_per_pix
        ->
    curvature / lateral error / heading error
        ->
    MPC

You do not need to remove `xm_per_pix` or `ym_per_pix`.

---

# Save after calibration

Save together:

- camera matrix `K`
- distortion coefficients `dist`
- BEV homography `H`
- BEV width and height
- `X_min`, `X_max`
- `Y_min`, `Y_max`
- `xm_per_pix`
- `ym_per_pix`
- image resolution used during calibration
- camera mounting position if measured

Treat these as one calibration set.

---

# Important assumption

This method assumes the road is approximately a flat plane.

That is usually a good assumption for a small BFMC competition track.

If the camera mounting, camera crop, resolution, or orientation changes, revalidate or repeat the calibration.

---

## Short version to remember

Do not calibrate the BEV only by visually adjusting a trapezoid until lanes look parallel.

Instead:

    calibrate lens
        ->
    undistort image
        ->
    measure known points on the real road
        ->
    assign real X/Y coordinates
        ->
    map them to desired metric BEV pixels
        ->
    calculate homography
        ->
    validate equal real distances near/middle/far
        ->
    save H + xm_per_pix + ym_per_pix

Final acceptance criterion:

The same real-world distance should represent approximately the same number of BEV pixels everywhere in the useful road region.
