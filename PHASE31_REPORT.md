# Phase 3.1: BFMC image preprocessing

## Files

Created:
- `src/algorithms/LaneDetection/utils/preprocessing.py`
- `src/algorithms/LaneDetection/utils/tuning.py`
- `validate_phase31.py`
- `PHASE31_REPORT.md`

Modified only `src/algorithms/LaneDetection/threads/threadLaneDetection.py` among existing source files. The original BFMC repository, both previous validators, Camera, Gateway, SerialHandler, message system, templates and requirements remain unchanged.

## Original code reused

Copied the original `PreprocessingMixin` perspective transform, Top-Hat helper, percentile statistics/floor, global thresholding, morphology and optional black-trapezoid mask without changing their mathematics. Omitted polynomial-guided thresholding and the unused ROI helper. Added `preprocess_frame_bgr()` to assemble the active initial/global path from the original thread's preprocessing calls, stopping before lane search.

`tuning.py` contains only constants referenced by these preprocessing operations. Their BFMC values are unchanged. No lane fits, geometry tracking, search, control or driving-speed constants were migrated.

## Pipeline and parameters

1. Convert the decoded 512x270 BGR frame with `cv2.COLOR_BGR2GRAY`.
2. Apply the original perspective transform at the same 512x270 output size. Source geometry uses top width 0.5, bottom width 0.95, bottom offset 0.0, horizon 0.71 and destination margins 0.0. For this resolution the source corners are `(13,270), (128,191), (384,191), (499,270)`; destination corners are `(0,270), (0,0), (512,0), (512,270)`. The original boundary coordinates, matrices and warp interpolation are preserved.
3. Optional car-front black-trapezoid suppression is retained but disabled, matching BFMC tuning.
4. Apply a 5x5 Gaussian blur followed by a rectangular 41x1 Top-Hat.
5. Use BFMC's global threshold path without previous fits. Compute the 90th percentile in the lower statistics region starting at 30% of BEV height. Ignore Top-Hat values below 3 when at least 100 remaining samples exist. Add offset -4, then enforce a minimum threshold of 15 and clip to 0..255. Compare the full Top-Hat image using `>=`.
6. Preserve morphology: optional opening is disabled; closing uses an 11x23 rectangular kernel for one iteration.
7. Return a `(270,512)` NumPy `uint8` binary image with values **0 and 1**, not 0 and 255. Stop here.

The original runtime leaves its ROI mask commented out; SCR also applies no ROI mask. CLAHE constants exist in the original tuning file but are not used by this active preprocessing implementation, so CLAHE was not added.

## Runtime behavior and safety

LaneDetection now inherits `PreprocessingMixin` alongside `ThreadWithStop` and `ImgCodecMixin`. After successful enabled-only decode and frame-shape validation it preprocesses the frame and checks output shape, dtype and 0/1 representation. The binary image is local to the current frame; it is not sent to any search or control stage.

The existing receiver/drain thread, newest-frame counter, duplicate/stale-frame skipping and disabled behavior are preserved. Preprocessing runs only after a new valid decoded frame while enabled. One preprocessing success diagnostic is printed per thread lifetime. Exceptions or invalid output are skipped safely; only the first failure in a consecutive episode is printed, and successful preprocessing resets suppression.

LaneControl and shutdown still publish only `{"steer": 0, "speed": 0, "valid": False}`. Preprocessing itself publishes no actuation.

## Validation performed

Run from SCRbrain with Python, NumPy and OpenCV available:

```text
python validate_phase1.py
python validate_phase2.py
python validate_phase31.py
```

`validate_phase31.py` runs the unchanged Phase 2 validator, which in turn runs the unchanged Phase 1 validator. All passed on Windows using spawned processes and a project-local ignored validation environment (Python 3.13, NumPy 2.5.3, OpenCV 5.0.0).

Verified:
- Source syntax/imports and all Phase 1/2 regression assertions.
- Road-like bright lines produce both background and foreground in a valid binary BEV.
- Perspective mapping and inverse mapping preserve the original source/destination geometry.
- Blank and changed-illumination images complete safely with valid output.
- Every copied tuning value matches the read-only BFMC reference.
- Binary outputs match the original BFMC global path exactly for all three deterministic images.
- Injected preprocessing failures remain quiet, recover after valid processing, and do not execute while disabled or emit actuation.
- Real synthetic JPEG/base64 -> Queue -> Gateway -> Pipe -> LaneDetection -> preprocessing integration reports first success exactly once.
- Real-process enable/disable, invalid-image recovery, zero-speed/invalid actuation and normal shutdown remain working.

The validator optionally compares against the adjacent BFMC reference when present; its core synthetic and integration tests do not require that repository or physical hardware.

No new physical camera or STM test was performed here. The user reports Phase 2 was already validated on the Raspberry Pi. Phase 3.1 still needs inspection on actual camera imagery to assess preprocessing quality; synthetic structural checks are not a quality benchmark or a reason to retune BFMC parameters.

## Boundary and next phase

Sliding-window lane search has NOT been implemented.
Polynomial fitting has NOT been implemented.
Lane geometry has NOT yet been extracted.
Steering calculation has NOT been implemented.
No valid movement command can be generated.

Phase 3.2 may later consume `binary_bev` for lane search. Previous-fit-guided thresholding can be reconnected only when real validated fits exist. No fake fits were created. Phase 3.1 ends here.
