# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import threading
import numpy as np
from src.algorithms.LaneDetection.utils.img_codec import ImgCodecMixin
from src.algorithms.LaneDetection.utils.preprocessing import PreprocessingMixin
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import serialCamera, LaneControl, LaneActuation
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber


class threadLaneDetection(ThreadWithStop, ImgCodecMixin, PreprocessingMixin):
    """Phase 3.1 preprocessing shell. No lane search or steering calculation."""
    def __init__(self, queuesList):
        super().__init__(pause=0.01)
        self.lane_enabled = False
        self.cameraSubscriber = messageHandlerSubscriber(queuesList, serialCamera, 'lastOnly', True)
        self.controlSubscriber = messageHandlerSubscriber(queuesList, LaneControl, 'fifo', True)
        self.laneActuationSender = messageHandlerSender(queuesList, LaneActuation)
        self._rx_lock = threading.Lock()
        self._rx_stop = threading.Event()
        self._latest_b64 = None
        # Count of frames received from the camera, incremented in the RX thread.
        self._rx_count = 0
        self._last_processed_rx_count = -1
        self._decode_error_active = False
        self._shape_error_active = False
        self._first_decoded_frame = True
        self._preprocessing_error_active = False
        self._first_preprocessed_frame = True
        self._rx_thread = threading.Thread(target=self._camera_rx_loop, daemon=True)

    def _camera_rx_loop(self):
        first = True
        try:
            # Drain even while disabled; retain at most one encoded frame.
            while not self._rx_stop.is_set():
                frame = self.cameraSubscriber.receive()
                if frame is None:
                    self._rx_stop.wait(0.001)
                    continue
                with self._rx_lock:
                    self._latest_b64 = frame
                    self._rx_count += 1
                if first:
                    print('[LANE] First camera frame received', flush=True)
                    first = False
        except (OSError, EOFError) as exc:
            print(f'[LANE][ERROR] Camera receiver failed: {exc}', flush=True)

    def safe_stop(self):
        self.laneActuationSender.send({'steer': 0, 'speed': 0, 'valid': False})

    def state_change_handler(self):
        message = self.controlSubscriber.receive()
        if message is None:
            return
        if not isinstance(message, dict) or type(message.get('enable')) is not bool:
            print('[LANE][ERROR] LaneControl requires a boolean enable', flush=True)
            self.lane_enabled = False
            self.safe_stop()
            return
        if message['enable'] and not self.lane_enabled:
            # Wait for a frame received after the disabled -> enabled transition.
            with self._rx_lock:
                self._last_processed_rx_count = self._rx_count
        self.lane_enabled = message['enable']
        state = 'enabled' if self.lane_enabled else 'disabled'
        print(f'[LANE] Lane following {state}', flush=True)
        # Enabling this shell never creates a movement command.
        self.safe_stop()

    def thread_work(self):
        if not self.lane_enabled:
            return
        with self._rx_lock:
            b64 = self._latest_b64
            rx_count = self._rx_count
        if b64 is None or rx_count == self._last_processed_rx_count:
            return
        self._last_processed_rx_count = rx_count
        frame_bgr = self._decode_b64_jpg_to_bgr(b64)
        if frame_bgr is None:
            if not self._decode_error_active:
                print('[LANE][ERROR] Camera frame decode failed', flush=True)
            self._decode_error_active = True
            return
        self._decode_error_active = False
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3
                or frame_bgr.shape != (270, 512, 3)):
            if not self._shape_error_active:
                print(f'[LANE][ERROR] Unexpected camera frame shape: '
                      f'{getattr(frame_bgr, "shape", None)}, expected (270, 512, 3)', flush=True)
            self._shape_error_active = True
            return
        self._shape_error_active = False
        if self._first_decoded_frame:
            print('[LANE] First camera frame decoded successfully: 512x270 BGR', flush=True)
            self._first_decoded_frame = False
        try:
            binary_bev = self.preprocess_frame_bgr(frame_bgr)
            if (not isinstance(binary_bev, np.ndarray)
                    or binary_bev.ndim != 2 or binary_bev.shape != (270, 512)
                    or binary_bev.dtype != np.uint8
                    or not np.all((binary_bev == 0) | (binary_bev == 1))):
                raise ValueError('Expected a 270x512 uint8 binary BEV with values 0/1')
        except Exception as exc:
            if not self._preprocessing_error_active:
                print(f'[LANE][ERROR] Frame preprocessing failed: {exc}', flush=True)
            self._preprocessing_error_active = True
            return
        self._preprocessing_error_active = False
        if self._first_preprocessed_frame:
            print('[LANE] First frame preprocessing completed successfully: '
                  '512x270 binary BEV', flush=True)
            self._first_preprocessed_frame = False
        # Phase 3.1 ends at binary_bev. No lane search or actuation here.

    def run(self):
        self._rx_thread.start()
        try:
            super().run()
        finally:
            self.lane_enabled = False
            self.safe_stop()
            self.cameraSubscriber.unsubscribe()
            self.controlSubscriber.unsubscribe()
            self._rx_stop.set()
            self._rx_thread.join(timeout=1)
