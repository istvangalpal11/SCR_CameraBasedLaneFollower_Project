"""Hardware-free compilation, import, protocol and spawned routing checks."""
import ast
import importlib
import multiprocessing as mp
from pathlib import Path
from queue import Full
import time


def validate():
    root = Path(__file__).parent
    for path in root.rglob('*.py'):
        source = path.read_text(encoding='utf-8-sig')
        compile(source, str(path), 'exec')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('src.'):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), (path, alias.name)
    importlib.import_module('main')  # Import must not start processes.
    from src.hardware.serialhandler.threads.messageconverter import MessageConverter
    converter = MessageConverter()
    assert converter.get_command('speed', speed=0) == '#speed:0;;\r\n'
    assert converter.get_command('steer', steerAngle=-12) == '#steer:-12;;\r\n'
    assert converter.get_command('kl', mode=0) == '#kl:0;;\r\n'

    # Exercise the SerialHandler boundary without opening any serial port.
    from src.hardware.serialhandler.threads.threadWrite import threadWrite
    class FakeSerial:
        def __init__(self):
            self.packets = []

        def write(self, data):
            self.packets.append(data)
            return len(data)

    class OneMessage:
        def __init__(self, value):
            self.value = value

        def receive(self):
            return self.value

    writer = threadWrite.__new__(threadWrite)
    writer.converter = converter
    writer.serialCon = FakeSerial()
    writer._first_command = True
    writer.subscriber = OneMessage({'steer': -12, 'speed': 80, 'valid': True})
    writer.thread_work()
    assert writer.serialCon.packets == [b'#steer:-12;;\r\n', b'#speed:80;;\r\n']
    writer.subscriber = OneMessage({'steer': 12, 'speed': 80, 'valid': False})
    writer.thread_work()
    assert writer.serialCon.packets[-2:] == [b'#speed:0;;\r\n', b'#kl:0;;\r\n']
    writer.subscriber = OneMessage({'steer': float('nan'), 'speed': 80, 'valid': True})
    writer.thread_work()
    assert writer.serialCon.packets[-2:] == [b'#speed:0;;\r\n', b'#kl:0;;\r\n']

    from src.gateway.processGateway import processGateway
    from src.algorithms.LaneDetection.processLaneDetection import processLaneDetection
    from src.utils.messages.allMessages import LaneControl, LaneActuation, serialCamera
    from src.utils.messages.messageHandlerSender import messageHandlerSender
    from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
    queues = {name: mp.Queue(maxsize=4 if name == 'General' else 0)
              for name in ('Critical', 'Warning', 'General', 'Config')}
    registered = mp.Event()
    observer = messageHandlerSubscriber(queues, LaneActuation, 'fifo', True)
    gateway = processGateway(queues, registered, mp.Event())
    lane = processLaneDetection(queues, mp.Event())
    processes = []
    try:
        for process in (gateway, lane):
            process.start()
            processes.append(process)
            assert process.ready_event.wait(5), 'Process startup timed out'
        assert registered.wait(5), 'Subscriptions not registered'
        camera = messageHandlerSender(queues, serialCamera)
        # Large encoded-frame stand-ins only test the pipe drain, not a camera.
        for _ in range(30):
            try:
                camera.send('frame' * 20000, block=False)
            except Full:
                pass
        control = messageHandlerSender(queues, LaneControl)
        for enabled in (True, False, True, False):
            control.send({'enable': enabled})
            deadline = time.monotonic() + 5
            value = None
            while value is None and time.monotonic() < deadline:
                value = observer.receive()
                time.sleep(0.01)
            assert value == {'steer': 0, 'speed': 0, 'valid': False}, value
        print('PASS: syntax, imports, STM packet format, Gateway routing, camera drain, start/stop', flush=True)
    finally:
        for process in reversed(processes):
            process.stop()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(2)
                raise AssertionError('Process failed clean shutdown')
            assert process.exitcode == 0, process.exitcode
        for queue in queues.values():
            queue.close()
            queue.cancel_join_thread()
    print('PASS: clean spawned-process shutdown', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    validate()
