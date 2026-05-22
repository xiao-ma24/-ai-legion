import uuid

class TaskManager:
    def __init__(self):
        self._states: dict[str, dict] = {}

    def create(self) -> str:
        task_id = str(uuid.uuid4())[:8]
        self._states[task_id] = {"progress": [], "final": "", "done": False}
        return task_id

    def push(self, task_id: str, html: str):
        if task_id in self._states:
            self._states[task_id]["progress"].append(html)

    def set_final(self, task_id: str, html: str):
        if task_id in self._states:
            self._states[task_id]["final"] = html

    def get_progress(self, task_id: str) -> str:
        state = self._states.get(task_id)
        if not state:
            return '<div class="error-card">任务不存在或已过期</div>'
        return "\n".join(state["progress"])

    def get_final(self, task_id: str) -> str:
        state = self._states.get(task_id)
        if not state:
            return ""
        return state["final"]

    def is_done(self, task_id: str) -> bool:
        state = self._states.get(task_id)
        return state["done"] if state else True

    def mark_done(self, task_id: str):
        if task_id in self._states:
            self._states[task_id]["done"] = True


task_manager = TaskManager()
