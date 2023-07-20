from one_graph_search import run_graph_search

if __name__ == '__main__':
    # задаю параметры
    num_edges = 5
    des_degree = 15
    des_cluster = 0.6
    des_num_nodes = max_graph_size = 30
    des_label_assort = 1
    cycle = False
    path = False
    dense = False
    star = False

    run_graph_search(dense=dense,cycle=cycle,path=path,star=star, size=16, num_edges=num_edges, des_degree=des_degree,
                                                                              des_cluster=des_cluster, des_num_nodes=des_num_nodes,
                                                                              des_label_assort=des_label_assort, visualize=True)

