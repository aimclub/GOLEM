from one_graph_search import run_graph_search
import pickle

if __name__ == '__main__':
    # задаю параметры

    des_num_nodes = max_graph_size = 310
    des_degree = 100
    num_edges = 5
    des_cluster = 0.4


    des_label_assort = 1
    cycle = False
    path = False
    dense = False
    star = False

    G_new = run_graph_search(dense=dense,cycle=cycle,path=path,star=star, size=16, num_edges=num_edges, des_degree=des_degree,
                                                                              des_cluster=des_cluster, des_num_nodes=des_num_nodes,
                                                                              des_label_assort=des_label_assort, visualize=True)
   # pickle.dump(G_new.to_undirected(), open('G_40_8.pickle', 'wb'))
