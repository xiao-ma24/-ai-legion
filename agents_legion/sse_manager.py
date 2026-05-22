import asyncio
import uuid


class SSEManager:
    """管理 SSE 事件队列：每个 task_id 对应一个 asyncio.Queue"""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}

    def create(self) -> str:
        task_id = str(uuid.uuid4())[:8]
        self._queues[task_id] = asyncio.Queue()
        return task_id

    async def push(self, task_id: str, event: str, data: str):
        if task_id in self._queues:
            await self._queues[task_id].put({"event": event, "data": data})

    async def close(self, task_id: str):
        await self.push(task_id, "done", "")
        await asyncio.sleep(0.5)
        self._queues.pop(task_id, None)

    async def events(self, task_id: str):
        q = self._queues.get(task_id)
        if not q:
            yield {"event": "error", "data": "<p>任务已过期</p>"}
            return
        while True:
            msg = await q.get()
            yield msg
            if msg["event"] == "done":
                break


sse_manager = SSEManager()
