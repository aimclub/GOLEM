from one_graph_search_k2 import run_graph_search
import pandas as pd
import random
import pickle

if __name__ == '__main__':
    # задаю параметры

    des_num_nodes = max_graph_size = 40# int(math.sqrt(700))
    des_degree = 5
    num_edges = 5
    des_cluster = .01
    des_asp  = 2

    des_label_assort = 1

    cycle = False
    path = False
    dense = False
    star = False


    def funad(nn):
        if nn == 16:
            return [2, 5, 10]
        if nn == 32:
            return [2, 5, 10, 15,20]
        if nn == 64:
            return [2, 5, 10, 15, 20,25,30]

    all_combinations = []
    for nn in [64]:
        for dd in [35]:#funad(nn):
            for cl in [0.1]:
                for asp in [2.5]:
                    all_combinations.append((nn,dd,cl,asp))

    #df = pd.read_csv('analysis_data_for_AAAI_workshop_1_3_edges.csv')
    #while 1>0:
    (nn,dd,cl,asp) = random.choice(all_combinations)
    des_cluster =cl
    des_num_nodes = nn
    des_degree = dd
    des_asp =asp
    print(des_num_nodes, des_cluster, des_asp, des_degree)
    #if len(df[(df['num nodes']==des_num_nodes) & (df['des cl']==des_cluster) & (df['des asp']==des_asp) & (df['des ad']==des_degree)])==0:
    G_new, time_after, time_before, act_cl, act_asp,act_ad = run_graph_search(dense=dense,cycle=cycle,path=path,star=star, size=16, num_edges=num_edges, des_degree=des_degree,
                                                                      des_cluster=des_cluster, des_num_nodes=des_num_nodes,
                                                              des_label_assort=des_label_assort, des_asp=des_asp, visualize=True)

        #new_row = pd.Series([des_num_nodes, des_cluster, des_asp, des_degree, act_cl, act_asp, act_ad,
        #                 time_after - time_before, False], index=df.columns)

        # df = pd.concat([df, new_row], ignore_index=True)
        #df = df.append(new_row, ignore_index=True)

        #df.to_csv('analysis_data_for_AAAI_workshop_1_3_edges.csv')

       # with open('G_' + str(des_num_nodes) + '_' + str(des_cluster) + '_' + str(des_asp) + '_' + str(
        #        des_degree) + '_1_3_edges.txt', 'a') as f:
         #   for edge in (G_new.edges()):
          #      f.write(str(edge[0]) + ',' + str(edge[1]) + '\n')

