import collections
from datetime import timedelta
from functools import partial
from wrapt_timeout_decorator import timeout

from typing import Optional, Union, List

import pandas as pd
import time
#import timeout_decorator

from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph_node import LinkedGraphNode

import matplotlib.pyplot as plt
import networkx as nx
from random import choice, random,randint, sample
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
import numpy as np
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
import random
from bench_graph_search import run_graph_search

if __name__ == '__main__':
    #df = pd.DataFrame(
       # columns=['dense','star','cycle','path', 'num edges','num nodes', 'avg degree', 'des cl', 'fact deg', 'fact cl', 'time'])
    #df.to_csv('results.csv')


    num_edges = 5
    des_degree = 25
    des_cluster = 0.8
    des_num_nodes = max_graph_size = 30
    des_label_assort = 1
    cycle = False
    path = False
    dense = False
    star = False


    #SINGLE and BATCH always switched on
    # other variants: cycle, path, dense, star
    print('before parameters')
    for des_num_nodes in [40,50,60]:
        for des_degree in range(10,des_num_nodes-10+1,10):
            for des_cluster in np.arange(round((des_degree-1)/(des_num_nodes-1),2),round((des_degree-1)/(des_num_nodes-1),2)+0.41, 0.1 ):
                if des_cluster<1:
                    for num_edges in [5,10,15,20]:#range(5,int(des_num_nodes/2)+1, 5):
                        for cycle in [False]:
                            for path in [False]:
                                for star in [False]:
                                    for dense in [False]:
                                        df = pd.read_csv('results.csv')
                                        df = df.drop(columns=['Unnamed: 0'])
                                        if len(df[(df['num nodes']==des_num_nodes) & (df['avg degree']==des_degree)&(df['des cl']==des_cluster)&(df['num edges']==num_edges)&(df['cycle']==cycle)&(df['dense']==dense)&(df['path']==path)&(df['star']==star)])==0:
                                            print('my params desired', des_num_nodes,des_degree, des_cluster,num_edges,cycle, path, star, dense)

                                            trial = 0
                                            while trial < 1:
                                                try:
                                                    df = run_graph_search(df, dense=dense,cycle=cycle,path=path,star=star, size=16, num_edges=num_edges, des_degree=des_degree,
                                                                              des_cluster=des_cluster, des_num_nodes=des_num_nodes,
                                                                              des_label_assort=des_label_assort, visualize=True)
                                                    print('yaaayyy')
                                                    df.to_csv('results.csv')
                                                    trial+=1
                                                except TimeoutError:
                                                    try:
                                                        with open('timeouts.txt','a') as f:
                                                            f.writelines(f'num nodes: {des_num_nodes}, num edges:{num_edges}, cl:{des_cluster}, deg:{des_degree}, dense: {dense}, cycle:{cycle}, path:{path}, star:{star} ')
                                                            print('noooo')
                                                    except:
                                                        pass



                        # Generate target graph that will be sought by optimizer
