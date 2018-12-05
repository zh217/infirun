import signal


class GracefulExit(Exception):
    pass


def signal_handler(signum, frame):
    raise GracefulExit()


def setup_signal_handlers(signals=(signal.SIGTERM, signal.SIGINT)):
    for s in signals:
        signal.signal(s, signal_handler)
