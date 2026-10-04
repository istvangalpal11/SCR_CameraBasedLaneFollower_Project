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
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import serialCamera, LaneControl, LaneActuation
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber


class threadLaneDetection(ThreadWithStop):
    """Phase 1 shell. No frame decoding or steering calculation."""
    def __init__(self, queuesList):
        super().__init__(pause=0.01)
        self.lane_enabled = False
        self.cameraSubscriber = messageHandlerSubscriber(queuesList, serialCamera, 'lastOnly', True)
        self.controlSubscriber = messageHandlerSubscriber(queuesList, LaneControl, 'fifo', True)
        self.laneActuationSender = messageHandlerSender(queuesList, LaneActuation)
        self._rx_lock = threading.Lock()
        self._rx_stop = threading.Event()
        self._latest_b64 = None
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
                if first:
                    print('[LANE] First camera frame received', flush=True)
                    first = False
        except (OSError, EOFError) as exc:
            print(f'[LANE][ERROR] Camera receiver failed: {exc}', flush=True)

    def safe_stop(self):
        self.laneActuationSender.send({'steer': 0, 'speed': 0, 'valid': False})

    def thread_work(self):
        message = self.controlSubscriber.receive()
        if message is None:
            return
        if not isinstance(message, dict) or type(message.get('enable')) is not bool:
            print('[LANE][ERROR] LaneControl requires a boolean enable', flush=True)
            self.lane_enabled = False
            self.safe_stop()
            return
        self.lane_enabled = message['enable']
        state = 'enabled' if self.lane_enabled else 'disabled'
        print(f'[LANE] Lane following {state}', flush=True)
        # Enabling this shell never creates a movement command.
        self.safe_stop()

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


