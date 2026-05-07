import asyncio

from app.inference.schemas import QueuedFrame


class WebSocketSessionManager:
    """Coordinates active inference WebSocket sessions."""

    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        self._queues: dict[str, asyncio.Queue[QueuedFrame]] = {}

    def create_queue(self, session_id: str) -> asyncio.Queue[QueuedFrame]:
        queue: asyncio.Queue[QueuedFrame] = asyncio.Queue(maxsize=self.queue_size)
        self._queues[session_id] = queue
        return queue

    def remove_queue(self, session_id: str) -> None:
        self._queues.pop(session_id, None)

    @staticmethod
    def put_latest(queue: asyncio.Queue[QueuedFrame], frame: QueuedFrame) -> None:
        if queue.full():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(frame)
