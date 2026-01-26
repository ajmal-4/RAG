from typing import List, Dict, Any

from app.core.logger import logger
from app.core.config import settings
from app.sandbox.runner import run_python_sandbox


class ChartAgent:

    def __init__(self):
        pass

    def execute_python_chart_code(self, python_code: str):
        chart = run_python_sandbox(python_code)
        with open("test.html", "w", encoding="utf-8") as f:
            f.write(chart["html"])
        print("Saved interactive chart to test.html")

