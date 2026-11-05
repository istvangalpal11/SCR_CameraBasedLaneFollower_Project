### TODO 1 – Validate FramePacketOrder on the Raspberry Pi

**Goal:** Determine whether `FramePacketOrder` is necessary for checking the order of incoming camera frames.

**Tasks:**
- Monitor and count rejected frames during extended continuous camera operation.
- Categorize rejection reasons: duplicate frames, out-of-order frames, and invalid timestamps.
- Test camera restarts to verify correct frame ordering behavior.
- Minimize performance overhead by using counters instead of continuous `print()` statements.

**Decision:** If no frame-ordering issues are detected during testing, consider simplifying or removing `FramePacketOrder`. Until then, keep the current implementation.