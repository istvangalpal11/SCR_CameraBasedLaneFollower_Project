from multiprocessing import Event, Queue, freeze_support
from src.gateway.processGateway import processGateway
from src.hardware.camera.processCamera import processCamera
from src.algorithms.LaneDetection.processLaneDetection import processLaneDetection
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.utils.messages.allMessages import LaneControl
from src.utils.messages.messageHandlerSender import messageHandlerSender


def shutdown_process(process):
    process.stop()
    process.join(timeout=5)
    if process.is_alive():
        print(f'[MAIN][ERROR] {process.name} did not stop; terminating', flush=True)
        process.terminate()
        process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join()


def main():
    print('[MAIN] Starting Lane Following', flush=True)
    #creating queues for inter-process communication, with a max size of 6 for the 'General' queue and unlimited size for others
    queueList = {name: Queue(maxsize=6 if name == 'General' else 0)
                 for name in ('Critical', 'Warning', 'General', 'Config')}
    subscriptions_ready = Event()
    #starting the processes for gateway, lane detection, serial handler, and camera, passing the queues and events for inter-process communication
    gateway = processGateway(queueList, subscriptions_ready, Event())
    lane = processLaneDetection(queueList, Event())
    serial = processSerialHandler(queueList, Event())
    camera = processCamera(queueList, Event())
    started = []
    sender = messageHandlerSender(queueList, LaneControl)
    lane_enabled = False
    try:
        for label, process in (('Gateway', gateway), ('LaneDetection', lane),
                               ('SerialHandler', serial)):
            process.start()
            started.append(process)
            if not process.ready_event.wait(10):
                raise RuntimeError(f'{label} startup timed out')
            print(f'[MAIN] {label} process started', flush=True)
        if not subscriptions_ready.wait(10):
            raise RuntimeError('Gateway subscription registration timed out')
        camera.start()
        started.append(camera)
        if not camera.ready_event.wait(10):
            raise RuntimeError('Camera startup timed out')
        print('[MAIN] Camera process started', flush=True)
        print('Commands: start | stop | status | quit', flush=True)
        while True:
            command = input('SCR> ').strip().lower()
            if command in ('start', 'stop'):
                lane_enabled = command == 'start'
                sender.send({'enable': lane_enabled})
                print(f'[MAIN] Lane following {command.upper()} requested', flush=True)
            elif command == 'status':
                print('Lane following: ' + ('ENABLED' if lane_enabled else 'DISABLED'), flush=True)
            elif command == 'quit':
                break
            elif command:
                print('[MAIN] Unknown command. Use start | stop | status | quit', flush=True)
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception as exc:
        print(f'[MAIN][ERROR] {exc}', flush=True)
    finally:
        print('[MAIN] Shutdown requested', flush=True)
        sender.send({'enable': False})
        print('[MAIN] Stopping processes...', flush=True)
        # Producers first, serial writes a final zero-speed/KL0 packet, Gateway last.
        for process in (camera, lane, serial, gateway):
            if process in started:
                shutdown_process(process)
        for queue in queueList.values():
            queue.close()
            queue.cancel_join_thread()
        print('[MAIN] Shutdown complete', flush=True)


if __name__ == '__main__':
    freeze_support()
    main()
