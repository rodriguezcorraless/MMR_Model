# -*- coding: utf-8 -*-
"""
@author: Edwin, Martin, Sancho
"""

import networkx as nx
import ndlib.models.ModelConfig as mc
from MultipleMajority import MultipleMajority
from bokeh.io import show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend


#Network Topology
#graph = nx.erdos_renyi_graph(1000, .8)
#graph = nx.dense_gnm_random_graph(1000, 5000)
graph = nx.watts_strogatz_graph(7500, 5, .2)
#graph = nx.random_regular_graph(5, 1000)
#graph = nx.barabasi_albert_graph(1000, 5)
#graph = nx.powerlaw_cluster_graph(1000, 5, 0.2)
#graph = nx.duplication_divergence_graph(1000, 0.2)
#model selection

bias = .5
model = MultipleMajority(graph, bias)

config = mc.Configuration()
config.add_model_parameter('fraction_Adopter', 0.3)
config.add_model_parameter('fraction_Rejector', 0.3)
config.add_model_parameter("q", 7)
model.set_initial_status(config)

#Simulation execution
iterations = model.iteration_bunch(2500)
trends = model.build_trends(iterations)

viz = DiffusionTrend(model, trends)
p = viz.plot(width=500, height=500)
show(p)
