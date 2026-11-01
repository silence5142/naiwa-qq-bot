"""按群保存共享问答，保留每条问题的发言者标识。"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


def format_user_content(user_id, question):
    return f"【群成员标识：{user_id}】\n{question}"


class ConversationMemory:
    def __init__(self, path, max_turns=30):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.path = Path(path)
        self.max_turns = max_turns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS turns ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "group_id TEXT NOT NULL, user_id TEXT NOT NULL, "
                "question TEXT NOT NULL, answer TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS turns_group "
                "ON turns (group_id, id)"
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def load(self, group_id):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT user_id, question, answer FROM turns "
                "WHERE group_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (group_id, self.max_turns),
            ).fetchall()
        messages = []
        for user_id, question, answer in reversed(rows):
            messages.extend([
                {"role": "user", "content": format_user_content(user_id, question)},
                {"role": "assistant", "content": answer},
            ])
        return messages

    def append(self, group_id, user_id, question, answer):
        # 插入和裁剪在同一事务中完成，只保留完整的一问一答。
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO turns (group_id, user_id, question, answer) "
                "VALUES (?, ?, ?, ?)",
                (group_id, user_id, question, answer),
            )
            connection.execute(
                "DELETE FROM turns WHERE group_id = ? "
                "AND id NOT IN (SELECT id FROM turns "
                "WHERE group_id = ? ORDER BY id DESC LIMIT ?)",
                (group_id, group_id, self.max_turns),
            )

    def clear(self, group_id):
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM turns WHERE group_id = ?",
                (group_id,),
            )
