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

import math
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import LaneActuation
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.hardware.serialhandler.threads.messageconverter import MessageConverter


class threadWrite(ThreadWithStop):
    def __init__(self, queuesList):
        super().__init__(pause=0.01)
        self.subscriber = messageHandlerSubscriber(queuesList, LaneActuation, 'fifo', True)
        self.converter = MessageConverter()
        self.serialCon = None
        self._first_command = True
        print('[SERIAL] Looking for STM serial device...', flush=True)
        try:
            import serial
            from serial.tools.list_ports import comports
            ports = [p for p in comports() if p.vid == 0x0483
                     or (p.device.startswith('/dev/ttyACM') and
                         p.vid != 0x1366 and 'SEGGER' not in (p.description or ''))]
            if not ports:
                raise RuntimeError('STM serial device was not found')
            ports.sort(key=lambda p: (p.vid != 0x0483, p.device))
            self.serialCon = serial.Serial(ports[0].device, 115200, timeout=0.1, write_timeout=0.5)
            print(f'[SERIAL] Connected: {ports[0].device}', flush=True)
            self.safe_stop()
        except Exception as exc:
            print(f'[SERIAL][ERROR] Serial initialization failed: {exc}', flush=True)

    def send_to_serial(self, action, **values):
        command = self.converter.get_command(action, **values)
        if command == 'error':
            raise ValueError(f'Invalid STM {action} command: {values}')
        if self.serialCon is not None:
            data = command.encode('ascii')
            if self.serialCon.write(data) != len(data):
                raise OSError('Incomplete serial write')

    def safe_stop(self):
        self.send_to_serial('speed', speed=0)
        self.send_to_serial('kl', mode=0)

    def thread_work(self):
        message = self.subscriber.receive()
        if message is None:
            return
        try:
            if not isinstance(message, dict) or type(message.get('valid')) is not bool:
                raise ValueError('LaneActuation requires steer, speed and boolean valid')
            steer, speed = message['steer'], message['speed']
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (steer, speed)):
                raise ValueError('Steer and speed must be finite numbers')
            if self._first_command:
                print('[SERIAL] First LaneActuation command received', flush=True)
                self._first_command = False
            if not message['valid']:
                self.safe_stop()
            else:
                # Validate both packets before transmitting either one.
                steer, speed = int(steer), int(speed)
                if (self.converter.get_command('steer', steerAngle=steer) == 'error'
                        or self.converter.get_command('speed', speed=speed) == 'error'):
                    raise ValueError('Command exceeds STM protocol limits')
                # Check this later: why both value is sent if speed in not changed
                self.send_to_serial('steer', steerAngle=steer)
                self.send_to_serial('speed', speed=speed)
        except Exception as exc:
            print(f'[SERIAL][ERROR] LaneActuation/transmission failed: {exc}', flush=True)
            try:
                self.safe_stop()
            except Exception as stop_exc:
                print(f'[SERIAL][ERROR] Safe stop failed: {stop_exc}', flush=True)

    def run(self):
        try:
            super().run()
        finally:
            try:
                self.safe_stop()
            except Exception as exc:
                print(f'[SERIAL][ERROR] Shutdown stop failed: {exc}', flush=True)
            if self.serialCon is not None:
                self.serialCon.close()
            self.subscriber.unsubscribe()
            print('[SERIAL] Connection closed', flush=True)


