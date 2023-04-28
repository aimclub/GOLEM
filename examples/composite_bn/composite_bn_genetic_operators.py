from composite_model import CompositeModel
from copy import deepcopy
from math import ceil
from random import choice, sample
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from itertools import chain
from ML import ML_models


def custom_crossover_exchange_edges(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):

    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges_1 = new_graph_first.operator.get_edges()
            edges_2 = new_graph_second.operator.get_edges()
            count = ceil(min(len(edges_1), len(edges_2))/2)
            choice_edges_1 = sample(edges_1, count)
            choice_edges_2 = sample(edges_2, count)
            
            for pair in choice_edges_1:
                new_graph_first.operator.disconnect_nodes(pair[0], pair[1], False)
            for pair in choice_edges_2:
                new_graph_second.operator.disconnect_nodes(pair[0], pair[1], False)  
            
            old_edges1 = new_graph_first.operator.get_edges()
            old_edges2 = new_graph_second.operator.get_edges()

            new_edges_2 = [[new_graph_second.get_nodes_by_name(str(i[0]))[0], new_graph_second.get_nodes_by_name(str(i[1]))[0]] for i in choice_edges_1]
            new_edges_1 = [[new_graph_first.get_nodes_by_name(str(i[0]))[0], new_graph_first.get_nodes_by_name(str(i[1]))[0]] for i in choice_edges_2]

            for pair in new_edges_1:
                if pair not in old_edges1:
                    new_graph_first.operator.connect_nodes(pair[0], pair[1])
            for pair in new_edges_2:
                if pair not in old_edges2:
                    new_graph_second.operator.connect_nodes(pair[0], pair[1])                                             
            
            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break
                                  
        if old_edges1 == new_graph_first.operator.get_edges() and new_edges_1!=[] and new_edges_1!=None:
            new_graph_first = deepcopy(graph_first)
        if old_edges2 == new_graph_second.operator.get_edges() and new_edges_2!=[] and new_edges_2!=None:
            new_graph_second = deepcopy(graph_second)
    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

def custom_crossover_exchange_parents_both(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):

    
    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            parents1 = []
            parents2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges = new_graph_second.operator.get_edges()
            flatten_edges = list(chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node2=choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1 = new_graph_first.get_nodes_by_name(str(selected_node2))[0]
                parents1=selected_node1.nodes_from
                

                if parents1:
                    for p in parents1:
                        new_graph_first.operator.disconnect_nodes(p, selected_node1, False)
                if parents2:
                    for p in parents2:
                        new_graph_second.operator.disconnect_nodes(p, selected_node2, False)

                old_edges1 = new_graph_first.operator.get_edges()
                old_edges2 = new_graph_second.operator.get_edges()

                if parents2!=[] and parents2!=None:
                    parents_in_first_graph=[new_graph_first.get_nodes_by_name(str(i))[0] for i in parents2]
                    for parent in parents_in_first_graph:
                        if [parent, selected_node1] not in old_edges1:
                            new_graph_first.operator.connect_nodes(parent, selected_node1)

                if parents1!=[] and parents1!=None:
                    parents_in_second_graph=[new_graph_second.get_nodes_by_name(str(i))[0] for i in parents1]
                    for parent in parents_in_second_graph:
                        if [parent, selected_node2] not in old_edges2:
                            new_graph_second.operator.connect_nodes(parent, selected_node2)            


            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break    
        
        if old_edges1 == new_graph_first.operator.get_edges() and parents2!=[] and parents2!=None:
            new_graph_first = deepcopy(graph_first)                
        if old_edges2 == new_graph_second.operator.get_edges() and parents1!=[] and parents1!=None:
            new_graph_second = deepcopy(graph_second)       

    except Exception as ex:
        print(ex)    
    return new_graph_first, new_graph_second

def custom_crossover_all_model(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):

        
    num_cros = 100
    try:
        for _ in range(num_cros):
            selected_node1=choice(graph_first.nodes)
            if selected_node1.nodes_from == None or selected_node1.nodes_from == []:
                continue
            
            selected_node2=graph_second.get_nodes_by_name(str(selected_node1))[0]           
            if selected_node2.nodes_from == None or selected_node2.nodes_from == []:
                continue            

            model1 = selected_node1.content['parent_model']
            model2 = selected_node2.content['parent_model']

            selected_node1.content['parent_model'] = model2
            selected_node2.content['parent_model'] = model1

            break

    except Exception as ex:
        print(ex)
    return graph_first, graph_second


# Структурные мутации
# задаем три варианта мутации: добавление узла, удаление узла, разворот узла
def custom_mutation_add_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(other_random_node)] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(random_node)])
            if nodes_not_cycling:
                other_random_node.nodes_from.append(random_node)
                ml_models = ML_models()
                other_random_node.content['parent_model'] = ml_models.get_model_by_children_type(other_random_node)                
                break

    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph
 

def custom_mutation_delete_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                random_node.nodes_from.remove(other_random_node)
                if not random_node.nodes_from:
                    random_node.content['parent_model'] = None                
                break
    except Exception as ex:
        print(ex) 
    return graph


def custom_mutation_reverse_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                random_node.nodes_from.remove(other_random_node)
                if not random_node.nodes_from:
                    random_node.content['parent_model'] = None                  
                other_random_node.nodes_from.append(random_node)
                ml_models = ML_models()
                other_random_node.content['parent_model'] = ml_models.get_model_by_children_type(other_random_node)                
                break         
    except Exception as ex:
        print(ex)  
    return graph


def custom_mutation_add_model(graph: CompositeModel, **kwargs):
    try:
        all_nodes = graph.nodes
        nodes_with_parents = [node for node in all_nodes if (node.nodes_from!=[] and node.nodes_from!=None)]
        if nodes_with_parents == []:
            return graph
        node = choice(nodes_with_parents)
        ml_models = ML_models()
        node.content['parent_model'] = ml_models.get_model_by_children_type(node)
    except Exception as ex:
        print(ex)  
    return graph

def mutation_set1(graph: CompositeModel, **kwargs):
    return custom_mutation_add_model(custom_mutation_add_structure(graph, **kwargs))

def mutation_set2(graph: CompositeModel, **kwargs): 
    return custom_mutation_add_model(custom_mutation_delete_structure(graph, **kwargs))
        
def mutation_set3(graph: CompositeModel, **kwargs):
    return custom_mutation_add_model(custom_mutation_reverse_structure(graph, **kwargs))

def cross_set(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):
    g11, g12 = custom_crossover_exchange_parents_both(graph_first, graph_second, max_depth)
    g21, g22 = custom_crossover_exchange_edges(g11, g12, max_depth)
    g31, g32 = custom_crossover_all_model(g21, g22, max_depth)
    return g31, g32