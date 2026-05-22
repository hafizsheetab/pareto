import argparse
import os

import config
import numpy as np
import uniqx as ux
from grid import Grid
from solver import run
from visualize import plot_snapshots
n_steps     = 100 
n = 32
grid = Grid(N=n)
print(f"\n{grid}\n")

mod, runtime_inputs = run(grid, n_steps=n_steps)
runtime_inputs.append("backend=compiled")
print("[main] module built — submitting to gateway…", flush=True)
gateway =  "api.oriqx.com:443"
api_key = "uxk_291c86ad1347b417e96d897d0d655a19"
print("API_KEYYYYYYY", api_key)
client = ux.connect(gateway, api_key=api_key)
result = ux.preflight(mod, client=client,)

import matplotlib.pyplot as plt
from collections import Counter
import graphviz

opt = result.recommended                       # or result[opt_idx], result.by_label("cpu+qpu")
assignments = opt.get("node_assignments") or {}

# A) hardware distribution across nodes
counts = Counter(assignments.values())
plt.bar(counts.keys(), counts.values(),
        color=[{"cpu":"#AED6F1","gpu":"#A9DFBF","qpu":"#D7BDE2","sim":"#F9E79F"}[k] for k in counts])
plt.title(f"Job partition — {opt['label']}")
plt.ylabel("# nodes")
plt.savefig("./test.png")
from uniqx.core.ir import parse_module   # or whatever the text→Module entry is
ir_text = result.lower(opt["_idx"])
print(type(ir_text))
      # execution.py:327, lazy lowering
# lowered = parse_module(ir_text)
# print(lowered.to_text())
# graphviz.Source(lowered.to_dot()).render("lowered", format="svg")