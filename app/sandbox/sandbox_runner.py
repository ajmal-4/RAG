import sys
import json
import traceback

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def main():
    try:
        code = sys.stdin.read()

        globals_dict = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
            },
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
        }

        locals_dict = {}

        exec(code, globals_dict, locals_dict)

        if "fig" not in locals_dict:
            raise Exception("Your code must assign the Plotly figure to variable `fig`")

        fig = locals_dict["fig"]

        # Convert to HTML (fully interactive)
        html = fig.to_html(include_plotlyjs="cdn", full_html=False)

        print(json.dumps({
            "success": True,
            "html": html
        }))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }))

if __name__ == "__main__":
    main()
