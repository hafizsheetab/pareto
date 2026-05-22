import argparse
import os

import config
import numpy as np
import uniqx as ux
from grid import Grid
from solver import run
from visualize import plot_snapshots
import graphviz


grid = Grid(N=32)

plot_snapshots(grid=grid, snapshots=[], save_path="./zero.png" )

