# Phase 0.2 — frame metadata, timing and replay

Implemented only Phase 0.2. After the architecture simplification, production packets have no `timestamp_source` or replay-mode field; recorded/simulated mode selection and replay clock filtering live only in the notebook adapter. No lane candidates, row scan, lane model, tracking, IMU, EKF, MPC or control logic was added. The Phase 0.1 morphology code and zero iteration defaults are unchanged.

## Files

Created:
- `src/utils/frame_packet.py`: plain-dict packet creation/validation, bounded session ordering, shared decode/preprocess orchestration, and bounded timing summaries.
- `notebooks/helpers/frame_replay.py`: natural-order simulated image replay and recorded JSON/JSONL metadata replay; no preprocessing algorithm copies.
- `notebooks/02_Lane_Perception_Development.ipynb`: separate development notebook, five-image demo, saved-image sequence configuration, metadata display, timing summary, and a nonimplemented next-phase placeholder.
- `tests/validate_phase02.py`: eight hardware-free tests covering requirements A–H, plus an explicitly labeled development-host timing sample.
- `reports/PHASE02_REPORT.md`: this report.

Modified:
- `src/hardware/camera/threads/threadCamera.py`: per-session UUID, capture counter, capture-return timestamp before image conversion/encoding, optional latency subtraction, publish timestamp, dict publication.
- `src/hardware/camera/processCamera.py`: forwards the optional, validated capture-latency offset to its thread.
- `src/utils/messages/allMessages.py`: `serialCamera.msgType` changes from `str` to `dict`; Owner/msgID/queue stay unchanged.
- `src/algorithms/LaneDetection/threads/threadLaneDetection.py`: validates/orders complete packets, atomically replaces the latest packet, retains the existing drain thread, and exposes the latest BEV with metadata and timing.
- `tests/validate_phase1.py`, `tests/validate_phase2.py`, `tests/validate_phase31.py`: update artificial camera fixtures to the dictionary packet contract. Existing assertions are retained. Rapid unit injections use deterministic `replay_synthetic` timestamps instead of this Windows clock's repeated coarse ticks.

Not modified: production preprocessing/tuning, the completed `phase31_preprocessing.ipynb`, Gateway, sender/subscriber helpers, serial/control code, templates, main or dependencies. The optional architecture v1.1 document was not found in the inspected project location.

## Live flow and timestamp accuracy

```text
capture_array('lores')
  -> immediately time.monotonic_ns() (capture return)
  -> existing YUV/BGR conversion and JPEG/base64
  -> frame dict -> existing Queue/Gateway/Pipe
  -> existing camera drain thread -> newest complete packet
  -> JPEG decode -> unchanged preprocess_frame_bgr()
  -> latest_preprocessed_frame: binary_bev + original identity/capture metadata
```

Required packet fields are `session_id`, `frame_id`, `t_capture_ns`, and `image_b64`. There is no production provenance or replay-mode field. Camera packets also include `t_capture_return_ns`, `capture_latency_offset_ns`, and `t_publish_ns`. The Gateway forwards them unchanged.

Camera thread construction generates a fresh UUID. Frame IDs begin at zero and increment after each successful capture return, including frames subsequently dropped by a full queue, so IDs are never reused within that session. Capture rate limiting remains **5 FPS**; actual frame intervals and sensor exposure rate are not inferred from that nominal setting.

`processCamera(..., capture_latency_offset_ns=0)` accepts a nonnegative integer offset. The default is zero and **uncalibrated**. The estimated capture time is capture-return minus that offset. Negative, noninteger and timestamp-underflow configurations are rejected. No latency correction value was invented; this is not an exact sensor exposure timestamp.

The receiver accepts nonempty session strings, nonnegative integer IDs/timestamps (excluding bool), nonempty string image payloads, and optional diagnostic timestamps. JPEG decoding remains separate, so a metadata-valid but malformed JPEG is handled by the existing quiet decode diagnostic. Duplicate/out-of-order IDs or timestamps do not replace the latest valid packet. A new valid session resets its ordering baseline; up to 16 retired session IDs are retained to reject late packets from recent old sessions. This history is bounded, not a permanent registry of all sessions.

The whole image/metadata dictionary is snapshotted and stored under the existing lock. While disabled, reception/draining continues without decoding. Start still waits for the next received frame. Only the newest packet is processed, and each receive count is attempted once. Original capture metadata is never replaced by processing times. Existing zero-speed invalid LaneActuation behavior is preserved.

## Results and diagnostics

The lane thread exposes `latest_preprocessed_frame` (one read-only-by-convention latest dictionary) and `latency_statistics()`. The result contains the unchanged capture identity/time metadata, `binary_bev`, receipt/start/preprocessing-start/completion timestamps. No LaneModel message was created.

All live diagnostics use the host monotonic clock. Metrics cover capture-to-publish, publish-to-receive, waiting before processing, preprocessing, processing including JPEG decode, and estimated frame age. Samples are bounded to 256. Median/p95 require at least five samples; sample count and host clock resolution are reported. A single summary is printed at thread shutdown, not per frame.

The notebook-only `replay_timing_sample()` adapter passes only same-host processing/wait timestamps to diagnostics; the production statistics helper has no replay branch. Recorded or simulated capture timestamps are never subtracted from the replay computer's current clock. This prevents fabricated ages from different boots/machines/timelines.

## Notebook usage

Open `notebooks/02_Lane_Perception_Development.ipynb` with the existing OpenCV/NumPy/Matplotlib development environment. No additional packages or environments were installed.

- `IMAGE_DIRECTORY = None`: five deterministic demo images in memory; no files generated.
- `IMAGE_DIRECTORY = Path(...)`: use a local sequence directory, including paths outside this repository.
- `REPLAY_MODE = 'simulated'`: natural numeric filename order; `ASSUMED_FPS` is configurable, default 5.0. Capture timestamps are `round(frame_index * 1e9 / ASSUMED_FPS)` and do not depend on processing speed.
- Each simulated replay call gets a fresh session unless `REPLAY_SESSION_ID` is supplied. Use separate calls/directories for separate sequences; filenames alone cannot establish sequence continuity or true capture rate.
- `REPLAY_MODE = 'recorded'`: set `RECORDED_METADATA_PATH` to a JSON/JSONL manifest. JSON can contain a record array, a `frames` array, or a single record. Each row supplies `image_path`, original session/frame ID, capture timestamp, with no provenance tag required. Relative paths resolve in IMAGE_DIRECTORY, or beside the manifest. Manifest order and metadata are preserved; malformed ordering is reported rather than replaced with invented timing.
- `EXPECTED_FRAME_SHAPE = (270, 512, 3)` enforces actual camera-sized input by default. Explicitly set it to None only for non-camera-size exploratory data; the helper does not resize/recalibrate those images and such data does not establish production geometry quality.
- `MAX_FRAMES` bounds work by default; `DISPLAY_LIMIT` and the last-20 metadata summary bound output/memory. The notebook never starts main, camera, Gateway or serial processes.

Example recorded row:

```json
{"image_path":"frame0125.jpg","session_id":"pi-session-original","frame_id":125,"t_capture_ns":583204510000,"t_publish_ns":583205000000}
```

No complete recorder, sensor synchronization or timestamp conversion between clocks is included.

## Validation results

`tests/validate_phase02.py`: **all eight tests passed**.

- Required packet fields/types, image decode, validated offset and actual camera-thread capture/clock/encode call order under mocks.
- Unique camera sessions, frame-ID gaps after queue drops, duplicate/out-of-order/malformed rejection and restart acceptance at frame ID zero.
- Actual subscriber drain/LastOnly behavior with 20 incoming packets; newest image and identity stay paired, sender dictionary mutations cannot corrupt the stored snapshot, and no duplicate processing occurs.
- Original metadata survives decode/preprocessing unchanged; output matches direct preprocessing pixel-for-pixel, using configured BEV dimensions, uint8 and only 0/1.
- Deterministic stage-clock tests verify timestamp ordering, nonnegative durations, bounded statistics, insufficient-sample behavior and replay clock-domain exclusions.
- Saved-image natural sorting, simulated FPS progression, new sequence sessions and recorded metadata preservation.
- Real Gateway process plus real LaneDetection thread verifies queue/pipe transport, safe start/stop and normal shutdown.
- Phase 0.1 still has no enable flag and both zero iteration defaults genuinely skip morphology.

`tests/validate_phase2.py`: **passed**, including its Phase 1 regression chain and spawned Gateway/LaneDetection process transport test. No regression assertions were removed.

`tests/validate_phase31.py`: its Phase 1/2 chain and regional-threshold checks passed, then it hit the **pre-existing** historical BFMC `THRESH_TOPHAT_BLUR_K` comparison failure. That unrelated assertion and accepted tuning were not changed. It is not claimed as a fully passing validator.

The new notebook's Python cells were executed headlessly in demo and recorded modes. Metadata, simulated progression, recorded IDs/times, binary BEV output and insufficient-sample reporting passed. Hash checks confirmed protected production preprocessing/tuning, Phase 0.1, old notebook, Gateway and message helpers remain byte-identical. Notebook magics were retained; headless checking executed Python cell contents without IPython directives.

## Development-host timing, not Pi performance

Test environment: Windows 11 build 26200, Python 3.12.10, existing external `scr-preprocessing` environment. Ten synthetic direct-call samples reported preprocessing median/p95 **0/0 ms**, and processing including JPEG decode median/p95 **0/~8.25 ms**. The reported monotonic clock resolution is **15.625 ms**, so these are quantized/sub-resolution measurements, NOT evidence of zero work or precise sub-millisecond latency. They cannot be used as Raspberry Pi performance estimates. The code prints this resolution caveat and will produce useful higher-resolution samples on hosts whose monotonic clock supports them.

Actual Pi capture-to-publish, transport latency, p95 under sustained camera load, and capture-return/exposure offset remain unmeasured. A minimum sample count does not compensate for coarse clock resolution.

## Future SensorTimestamp and hardware checks

Picamera2 completed requests expose `get_metadata()`, allowing future code to obtain image data and metadata from the same request. This was inspected in the [official Picamera2 request implementation](https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py). Separately fetching another frame's metadata would risk mismatching its image.

The [official libcamera SensorTimestamp definition](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/control_ids_core.yaml) describes first-row exposure timing in the CLOCK_BOOTTIME domain. Compatibility/conversion to Python's monotonic clock must be verified on the actual Pi before using it. Phase 0.2 does not switch to sensor timing. Future recordings must retain their original times; the notebook must not compare them to the replay host clock as live ages.

Remaining hardware checks: Picamera2/libcamera initialization and actual frame rate/intervals, UUID/counter behavior across real restarts, real timestamp monotonicity across Pi processes, measured capture-return offset/uncertainty, USB/CPU load and sustained latency percentiles. No physical camera or STM was tested. No exposure calibration, IMU sampling/reception timestamps, fusion or control was implemented.

Phase 0.2 stops here. Row-scan and all later perception/control work remain unimplemented.
