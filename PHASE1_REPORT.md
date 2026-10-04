# SCR Phase 1 change report

Only the process/thread/message skeleton is implemented. The old `../brain` project was read as reference and was not modified. No lane-processing utilities or steering algorithm were migrated.

## A. Created files

All 32 new files are listed below. Every `__init__.py` is an empty package marker.

```text
SCRbrain/
  main.py
  requirements.txt
  validate_phase1.py
  PHASE1_REPORT.md
  src/
    __init__.py
    templates/
      __init__.py
      workerprocess.py
      threadwithstop.py
    gateway/
      __init__.py
      processGateway.py
      threads/
        __init__.py
        threadGateway.py
    hardware/
      __init__.py
      camera/
        __init__.py
        processCamera.py
        threads/
          __init__.py
          threadCamera.py
      serialhandler/
        __init__.py
        processSerialHandler.py
        threads/
          __init__.py
          threadWrite.py
          messageconverter.py
    algorithms/
      __init__.py
      LaneDetection/
        __init__.py
        processLaneDetection.py
        threads/
          __init__.py
          threadLaneDetection.py
    utils/
      __init__.py
      messages/
        __init__.py
        allMessages.py
        messageHandlerSender.py
        messageHandlerSubscriber.py
```

## B. Copied almost unchanged

- `threadwithstop.py`: exact BFMC copy.
- `messageHandlerSubscriber.py`: exact BFMC copy, retaining Config-queue subscription requests and FIFO/LastOnly pipe delivery.
- `messageHandlerSender.py`: original envelope and queue selection; optional nonblocking publishing added for bounded camera traffic.
- `workerprocess.py`: original lifecycle, readiness events, pause/resume, thread work hooks and stop events retained. Small additions handle startup exceptions, joining unstarted threads, and consistent diagnostic prefixes.

Original copyright/license headers are retained on copied and adapted BFMC components.

## C. Substantially simplified

- `main.py`: four processes, startup readiness, terminal commands, ordered stop/join and terminate/kill fallback. `status` reports the locally requested enable state, not confirmed hardware readiness.
- Gateway: original Owner/msgID routing and payload envelope retained; unrelated diagnostics removed. Registration runs before routing, and `get_nowait()` replaces unreliable cross-process `Queue.empty()` checks.
- Camera: only `serialCamera` publishing. Original PiCamera2 preview configuration, 512 x 270 lores conversion, JPEG/base64 encoding and 5 FPS limit retained. The main stream remains configured for compatibility but is neither captured nor published. Recording and configuration controls removed. Import/init/capture errors disable capture with one diagnostic rather than endless retries.
- LaneDetection: only enable state, control subscription, encoded-frame receiver/drain thread and safe-stop publication. The receiver stays active while disabled and retains at most one frame; no decoding or algorithm runs. Every enable/disable request publishes zero speed with `valid=False`.
- SerialHandler: only STM connection, LaneActuation consumption and motor/stop transmission. Connects once at 115200 baud with bounded writes. Prefers STMicroelectronics USB devices and retains a Linux ttyACM fallback excluding SEGGER. Removed sensor handling, telemetry, histories, reconnect subsystem and competition features.
- `messageconverter.py`: original formatting and integer/digit validation retained; command table restricted to speed, steer and KL (safe power-off). Packets remain `#speed:0;;\r\n`, `#steer:-12;;\r\n`, and `#kl:0;;\r\n`.
- Message catalog: only `serialCamera`, `LaneControl`, and `LaneActuation`. Camera and actuation Owner/IDs remain familiar; LaneControl producer is now main. Control and actuation use Critical priority so camera traffic cannot take precedence over stopping.

## D. Message flow and shutdown

```text
main          -- LaneControl   --> Queue --> Gateway --> Pipe --> LaneDetection
Camera        -- serialCamera  --> Queue --> Gateway --> Pipe --> LaneDetection
LaneDetection -- LaneActuation --> Queue --> Gateway --> Pipe --> SerialHandler --> STM
```

There are no direct data links between these modules. Shared events only coordinate startup/termination. Main waits for all three required Gateway subscriptions before starting Camera and accepting terminal commands.

`start` enables the lane shell but never moves the car in this phase. `stop` disables it and causes an invalid zero-speed actuation. SerialHandler maps invalid or malformed actuation to zero speed and KL0. On quit/EOF/Ctrl+C, main requests disable, stops Camera and LaneDetection, then SerialHandler, and finally Gateway. LaneDetection publishes a final safe stop; SerialHandler independently writes zero speed/KL0 before closing. A disconnected STM cannot receive a stop packet, and transmission failures are reported.

The General queue holds at most four waiting messages. Camera drops new frames if this queue is full. LaneDetection's dedicated receiver continuously drains the camera pipe and replaces its latest encoded frame. The Gateway retains BFMC's blocking pipe sends; a dead subscriber can still require the bounded shutdown fallback.

## Validation

Run from `SCRbrain` with a working Python installation:

```text
python validate_phase1.py
python main.py
```

The hardware-free validation compiles every Python source, checks project imports and imported symbols, confirms importing main does not start processes, checks exact STM packets, tests SerialHandler with an in-memory serial stand-in, and exercises real spawned Gateway/LaneDetection processes. Large string frame stand-ins test drainage and routing responsiveness during repeated start/stop requests. Processes must join normally with exit code zero.

Checks performed on Windows:

- Syntax/import and spawned Gateway/LaneDetection routing/drain/shutdown checks passed.
- Exact packet formatting and mocked SerialHandler valid/invalid/malformed-command stop checks passed.
- Four-process main entry-point smoke check with `start`, `status`, `stop`, `status`, `quit` passed; missing Picamera2/pyserial produced clear diagnostics, and shutdown completed normally.
- Source scan found no removed-feature references, forbidden directories, lane-processing imports, direct camera-to-lane communication, or lane serial access.
- Multiprocessing entry protection is present in main and the validation script.

The bundled MySQL Python interpreter was used with its DLL directory on the command-local PATH. Python is not otherwise available on this shell's PATH. No dependencies or hardware were installed or tested.

## E. Hardware-dependent items

- Confirm PiCamera2/libcamera availability, preview stream compatibility, lores YUV format, capture and JPEG/base64 delivery on the actual Raspberry Pi.
- Confirm STM USB selection, access permissions, baud rate, accepted steering/speed units and bounds, and zero-speed/KL0 firmware behavior.
- Verify actual stop timing and camera/serial cleanup on hardware, including disconnects and blocked I/O. Software packet tests cannot confirm physical stopping.
- On Raspberry Pi OS install `python3-picamera2` and `python3-opencv` with apt; a venv should expose system packages. Install `pyserial` from requirements. Check package compatibility on the target OS.

## F. Next-phase TODOs — not implemented

- Migrate the original image decoder, preprocessing, thresholding and BEV utilities separately.
- Migrate lane search, fitting, confidence and steering computation only after approval.
- Define the constant operating speed and validated actuation contract/units.
- Verify the firmware's required KL enable sequence before allowing future nonzero commands; Phase 1 only sends KL0 for safe stop.
- Validate the complete real Camera -> Gateway -> LaneDetection -> Gateway -> SerialHandler -> STM path.

Phase 1 ends here.
