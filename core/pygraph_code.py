from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
import  pycallgraph

#pycallgraph.start_trace(reset = False)
import sys
sys.path.append(".")
sys.path.append("..")


config = Config(max_depth=1)
config.trace_filter = GlobbingFilter(exclude=[  'pycallgraph.*', './.venv/*', './.venv/*'],include=['./core/*'])

graphviz = GraphvizOutput(output_file='filter_max_depth.png')
with PyCallGraph(output=graphviz, config=config):
    from .examples import cart_pole

    #import .examples.cart_pole
    