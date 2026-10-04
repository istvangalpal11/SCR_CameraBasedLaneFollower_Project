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

from multiprocessing import Process, Event

class WorkerProcess(Process):
    def __init__(self, queuesList, ready_event=None, daemon=True):
        """WorkerProcess is an abstract class for description a general structure and interface a process.

        Parameters
        ----------
        inPs : list(Pipe)
            input pipes
        outPs : list(Pipe)
            output pipes
        ready_event : multiprocessing.Event, optional
            event to signal when threads are ready
        daemon : bool, optional
            daemon process flag, by default True
        """
        super(WorkerProcess, self).__init__()

        self.queuesList = queuesList
        self.ready_event = ready_event

        self.daemon = daemon
        self.threads = list()
        self.log_prefix = {
            "processGateway": "GATEWAY", "processCamera": "CAMERA",
            "processLaneDetection": "LANE", "processSerialHandler": "SERIAL",
        }.get(self.__class__.__name__, self.__class__.__name__)

        # Inter-process communication for pause/resume
        self._pause_event = Event()
        self._resume_event = Event()
        
        # Intra-process coordination
        self._blocker = Event()
        self._hard_kill = Event()
        
    def _init_threads(self):
        """It initializes the threads of the process and adds the thread to the 'threads' list, which will be automatically started and stopped in the 'run' method.

        Raises
        ------
        NotImplementedError
            Have to implement the initialization of threads
        """
        raise NotImplementedError

    def run(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        """This method applies the initialization of the theards and starts all of them. The process ignores the keyboardInterruption signal and can terminate by applying the 'stop' method.
        The process will be blocked, until an other process use the 'stop' function. After appling the function it terminates all subthread.
        """
        try:
            self._init_threads()
            for th in self.threads:
                th.daemon = self.daemon
                th.start()
        except Exception as exc:
            print(f"[{self.log_prefix}][ERROR] Startup failed: {exc}", flush=True)
            self.stop_threads()
            return
            
        # Signal that threads are ready
        if self.ready_event:
            self.ready_event.set()
            
        # wait to set internal flag true for the event
        while not self._blocker.is_set():
            if getattr(self, "_hard_kill", None) and self._hard_kill.is_set():
                break
            try:
                # handle the state change
                self.state_change_handler()

                # do the actual work
                self.process_work()
        
                # check for pause/resume commands
                if self._pause_event.is_set():
                    for thread in self.threads:
                        thread.pause()
                    self._pause_event.clear()
                    
                if self._resume_event.is_set():
                    for thread in self.threads:
                        thread.resume()
                    self._resume_event.clear()
                    
                self._blocker.wait(0.1) # shorter wait for responsiveness
            except KeyboardInterrupt:
                print(f"[{self.log_prefix}] KeyboardInterrupt received", flush=True)
                self._blocker.set()
                
        # cleanup section
        print(f"[{self.log_prefix}] Cleaning up threads...", flush=True)
        self.stop_threads()
        print(f"[{self.log_prefix}] Process stopped", flush=True)

    def stop_threads(self):
        for th in self.threads:
            if hasattr(th, "stop") and callable(getattr(th, "stop")):
                # resume thread first if it's paused
                if th.is_paused():
                    th.resume()

                th.stop()
                if th.ident is not None:
                    th.join(timeout=3)

                if th.is_alive():
                    print(f"[{self.log_prefix}][ERROR] Thread {th.name} is blocked", flush=True)
            else:
                print(f"[{self.log_prefix}][ERROR] Thread {th.name} has no stop function", flush=True)
                
    def state_change_handler(self):
        """This method is called to handle the state change of the process. It will be overridden by the child process."""
        pass

    def process_work(self):
        """This method is called when the process is running. It will be overridden by the child process."""
        pass

    def pause_threads(self):
        """Signal this process to pause its threads (called from main process)."""
        self._pause_event.set()

    def resume_threads(self):
        """Signal this process to resume its threads (called from main process)."""
        self._resume_event.set()

    def are_threads_paused(self):
        """Check if any threads are currently paused."""
        for thread in self.threads:
            if thread.is_paused():
                return True
        return False

    def stop(self):
        """Proper stop for the process"""
        print(f"[{self.log_prefix}] Stop requested", flush=True)
        # Resume threads if paused
        self._hard_kill.set()
        self.resume_threads()
        # Trigger exit loop
        self._blocker.set()
    
