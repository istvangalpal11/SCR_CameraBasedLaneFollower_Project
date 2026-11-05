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
import time
from src.utils.frame_packet import (FramePacketOrder, FrameDecodeError, FrameShapeError,
                                    preprocess_frame_packet, LatencyStatistics)
import numpy as np
from src.algorithms.LaneDetection.utils.img_codec import ImgCodecMixin
from src.algorithms.LaneDetection.utils.preprocessing import PreprocessingMixin
from src.algorithms.LaneDetection.utils import tuning as lane_tuning
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
        self._latest_packet = None
        self._packet_order = FramePacketOrder()
        self._packet_error_active = False
        self.latest_preprocessed_frame = None
        self._timing_stats = LatencyStatistics()
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
                if not self._store_camera_packet(frame, time.monotonic_ns()):
                    continue
                if first:
                    print('[LANE] First camera frame received', flush=True)
                    first = False
        except (OSError, EOFError) as exc:
            print(f'[LANE][ERROR] Camera receiver failed: {exc}', flush=True)

    def _store_camera_packet(self, packet, t_received_ns=None):
        """Validate and replace the entire latest packet atomically; never store fields separately."""
        received = time.monotonic_ns() if t_received_ns is None else t_received_ns
        try:
            from src.utils.frame_packet import validate_frame_packet
            packet = validate_frame_packet(packet)
            if max(packet['t_capture_ns'], packet.get('t_publish_ns', 0)) > received:
                raise ValueError('Live frame timestamp is ahead of the receiving monotonic clock')
            with self._rx_lock:
                packet = self._packet_order.accept(packet)
                self._latest_packet = dict(packet, t_received_ns=received)
                self._rx_count += 1
            self._packet_error_active = False
            return True
        except ValueError as exc:
            if not self._packet_error_active:
                print(f'[LANE][ERROR] Invalid camera packet: {exc}', flush=True)
            self._packet_error_active = True
            return False

    def latency_statistics(self):
        with self._rx_lock:
            return self._timing_stats.summary()

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
            packet = self._latest_packet
            rx_count = self._rx_count
        if packet is None or rx_count == self._last_processed_rx_count:
            return
        self._last_processed_rx_count = rx_count
        try:
            result = preprocess_frame_packet(packet, self, t_received_ns=packet['t_received_ns'])
        except FrameDecodeError:
            if not self._decode_error_active:
                print('[LANE][ERROR] Camera frame decode failed', flush=True)
            self._decode_error_active = True
            return
        except FrameShapeError as exc:
            self._decode_error_active = False
            if not self._shape_error_active:
                print(f'[LANE][ERROR] {exc}', flush=True)
            self._shape_error_active = True
            return
        except Exception as exc:
            if not self._preprocessing_error_active:
                print(f'[LANE][ERROR] Frame preprocessing failed: {exc}', flush=True)
            self._preprocessing_error_active = True
            return
        self._decode_error_active = self._shape_error_active = self._preprocessing_error_active = False
        with self._rx_lock:
            self.latest_preprocessed_frame = result
            self._timing_stats.add(result)
        if self._first_decoded_frame:
            print('[LANE] First camera frame decoded successfully: 512x270 BGR', flush=True)
            self._first_decoded_frame = False
        if self._first_preprocessed_frame:
            print('[LANE] First frame preprocessing completed successfully: '
                  f'{lane_tuning.BEV_OUTPUT_WIDTH}x{lane_tuning.BEV_OUTPUT_HEIGHT} binary BEV', flush=True)
            self._first_preprocessed_frame = False
        # One latest BEV+metadata result only. No lane search or actuation here.

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
            print(f'[LANE] Host-monotonic timing summary: {self.latency_statistics()}', flush=True)
