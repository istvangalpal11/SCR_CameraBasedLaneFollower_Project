### TODO 1 – Validate FramePacketOrder on the Raspberry Pi

**Goal:** Determine whether `FramePacketOrder` is necessary for checking the order of incoming camera frames.

**Tasks:**
- Monitor and count rejected frames during extended continuous camera operation.
- Categorize rejection reasons: duplicate frames, out-of-order frames, and invalid timestamps.
- Test camera restarts to verify correct frame ordering behavior.
- Minimize performance overhead by using counters instead of continuous `print()` statements.

**Decision:** If no frame-ordering issues are detected during testing, consider simplifying or removing `FramePacketOrder`. Until then, keep the current implementation.


### TODO 2 – Merge SCR Lane Detection into the Original BFMC Code

**Goal:** Integrate the current SCR Lane Detection implementation into the original BFMC organizers' codebase when the physical car becomes available.

The original BFMC code does not contain our Lane Detection module, so copying the `algorithms/LaneDetection` directory should not cause major conflicts.

However, the Phase 0.2 implementation also modified shared camera and messaging components. These changes must be carefully merged into the original BFMC code.

**Required merge tasks:**
- **Camera (`threadCamera.py`, `processCamera.py`):** Add frame timestamps using `time.monotonic_ns()`, configurable capture-latency offset, `session_id`, `frame_id`, and dictionary-based frame transmission instead of sending only a base64 string.
- **Messages (`allMessages.py`):** Update the `serialCamera` message type from `str` to `dict`, preserving its existing routing configuration.
- **Utilities (`frame_packet.py`):** Transfer the shared frame packet validation, ordering, preprocessing orchestration, and latency statistics.
- **LaneDetection (`threadLaneDetection.py`):** Ensure it receives complete frame packets, preserves metadata, and maintains the existing `lastOnly` behavior.
- **Compatibility:** Check other camera-message consumers and adapt them if necessary. Preserve the original BFMC camera configuration and hardware functionality.

**Available references during merging:**
1. Current SCR project with Phase 0.2 modifications.
2. Previous BFMC project code.
3. Original BFMC organizers' code.

Compare all three implementations to identify the required changes and avoid overwriting unrelated functionality.

**Validation:** Test camera capture, frame transmission, preprocessing, timestamps, and safe operation on the actual Raspberry Pi before proceeding.