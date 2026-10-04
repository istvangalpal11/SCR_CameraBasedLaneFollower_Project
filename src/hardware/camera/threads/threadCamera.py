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

import base64
import time
from queue import Full
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import serialCamera
from src.utils.messages.messageHandlerSender import messageHandlerSender


class threadCamera(ThreadWithStop):
    def __init__(self, queuesList):
        super().__init__()
        self.sender = messageHandlerSender(queuesList, serialCamera)
        self.camera = None
        self.frame_rate = 5
        self._next_frame_t = time.monotonic()
        print('[CAMERA] Initializing camera...', flush=True)
        try:
            from picamera2 import Picamera2
            import cv2
            self.cv2 = cv2
            if not Picamera2.global_camera_info():
                raise RuntimeError('No camera detected')
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                buffer_count=1, queue=False,
                main={'format': 'RGB888', 'size': (2048, 1080)},
                lores={'size': (512, 270)}, encode='lores')
            self.camera.configure(config)
            self.camera.start()
            print('[CAMERA] Camera initialized successfully', flush=True)
        except Exception as exc:
            print(f'[CAMERA][ERROR] Camera initialization failed: {exc}. '
                  'Camera requires Raspberry Pi OS/Picamera2 and OpenCV.', flush=True)
            self._close_camera()

    def _close_camera(self):
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception as exc:
                print(f'[CAMERA][ERROR] Camera close failed: {exc}', flush=True)
            self.camera = None

    def thread_work(self):
        if self.camera is None:
            self._blocker.wait(0.2)
            return
        now = time.monotonic()
        if now < self._next_frame_t:
            self._blocker.wait(self._next_frame_t - now)
            return
        self._next_frame_t = now + 1 / self.frame_rate
        try:
            frame = self.camera.capture_array('lores')
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_YUV2BGR_I420)
            ok, encoded = self.cv2.imencode('.jpg', frame)
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            self.sender.send(base64.b64encode(encoded).decode('utf-8'), block=False)
        except Full:
            pass  # Drop a frame rather than accumulate a camera backlog.
        except Exception as exc:
            print(f'[CAMERA][ERROR] Capture failed; camera disabled: {exc}', flush=True)
            self._close_camera()

    def run(self):
        try:
            super().run()
        finally:
            self._close_camera()


