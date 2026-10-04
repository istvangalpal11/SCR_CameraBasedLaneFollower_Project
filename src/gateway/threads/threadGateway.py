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

from queue import Empty
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import serialCamera, LaneControl, LaneActuation


class threadGateway(ThreadWithStop):
    def __init__(self, queuesList, subscriptions_ready):
        super().__init__()
        self.queuesList = queuesList
        self.sendingList = {}
        self.subscriptions_ready = subscriptions_ready
        self.messages = {(m.Owner.value, m.msgID.value): m.__name__
                         for m in (serialCamera, LaneControl, LaneActuation)}

    def subscribe(self, message):
        key = (message['Owner'], message['msgID'])
        receiver = message['To']['receiver']
        self.sendingList.setdefault(key, {})[receiver] = message['To']['pipe']
        print(f"[GATEWAY] Subscriber registered: {receiver} -> {self.messages[key]}", flush=True)
        if all(self.sendingList.get(k) for k in self.messages):
            self.subscriptions_ready.set()

    def unsubscribe(self, message):
        key = (message['Owner'], message['msgID'])
        pipe = self.sendingList.get(key, {}).pop(message['To']['receiver'], None)
        if pipe is not None:
            pipe.close()

    def send(self, message):
        key = (message['Owner'], message['msgID'])
        payload = {'Type': message['msgType'], 'value': message['msgValue'],
                   'id': key[1], 'Owner': key[0]}
        for receiver, pipe in list(self.sendingList.get(key, {}).items()):
            try:
                pipe.send(payload)
            except (OSError, EOFError) as exc:
                print(f"[GATEWAY][ERROR] {key} -> {receiver}: {exc}", flush=True)
                pipe.close()
                del self.sendingList[key][receiver]

    def thread_work(self):
        try:
            # Registration first; Queue.empty() is unreliable across processes.
            for _ in range(16):
                try:
                    message = self.queuesList['Config'].get_nowait()
                except Empty:
                    break
                action = message['Subscribe/Unsubscribe'].lower()
                if action == 'subscribe':
                    self.subscribe(message)
                elif action == 'unsubscribe':
                    self.unsubscribe(message)
            for name in ('Critical', 'Warning', 'General'):
                try:
                    message = self.queuesList[name].get_nowait()
                except Empty:
                    continue
                self.send(message)
                break
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[GATEWAY][ERROR] Invalid message: {exc}", flush=True)

    def stop(self):
        super().stop()


