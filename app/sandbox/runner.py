import subprocess
import tempfile
import shutil
import json
import os
import sys

def run_python_sandbox(code: str, timeout_sec=20):
    temp_dir = tempfile.mkdtemp()

    try:
        sandbox_file = os.path.join(os.path.dirname(__file__), "sandbox_runner.py")

        proc = subprocess.Popen(
            [sys.executable, "-I", sandbox_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=temp_dir,
            text=True
        )

        try:
            stdout, stderr = proc.communicate(code, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {"success": False, "error": "Execution timed out"}

        if stderr:
            return {"success": False, "error": stderr}

        try:
            result = json.loads(stdout.strip().splitlines()[-1])
        except:
            return {"success": False, "error": "Invalid sandbox output", "raw": stdout}

        return result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# -------------------------
# TEST
# -------------------------
if __name__ == "__main__":
    test_code = """
data = [
    {"month": "Jan", "sales": 120},
    {"month": "Feb", "sales": 180},
    {"month": "Mar", "sales": 150}
]

df = pd.DataFrame(data)

fig = px.line(df, x="month", y="sales", title="Sales Trend")
"""

    res = run_python_sandbox(test_code)

    if res["success"]:
        with open("test.html", "w", encoding="utf-8") as f:
            f.write(res["html"])
        print("Saved interactive chart to test.html")
    else:
        print("Error:", res)
