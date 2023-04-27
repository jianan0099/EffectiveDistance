import json
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import *


def get_arr(dict_, node_list):
    arr = np.zeros(len(node_list))
    for i in range(len(node_list)):
        c = node_list[i]
        arr[i] = dict_[c]
    return arr


def init_data():
    # get_initial_data
    with open('final_data/common.json', 'r') as f:
        common_dict = json.load(f)
    common_ = common_dict['common']
    node_num_ = len(common_)
    with open('final_data/popu_without_1000.json', 'r') as f:
        popu_final = json.load(f)
    Popu_ = get_arr(popu_final, common_) * 1000
    G_air_ = nx.read_gexf('final_data/G_air_2019_sample_complete.gexf').subgraph(common_)
    return node_num_, common_, Popu_, G_air_


def get_source_node_neighbor_weight_map(G_):
    neighbor_weight_map = defaultdict(float)
    for node_ in ['CHN', 'HKG', 'MAC']:
        for nbr in G_[node_]:
            if nbr not in ['CHN', 'HKG', 'MAC'] and node_ in G_[nbr]:
                neighbor_weight_map[nbr] += G_[node_][nbr]['weight'] + G_[nbr][node_]['weight']
    sorted_neighbor_weight_map = sorted(neighbor_weight_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_neighbor_weight_map


def get_top_edges_from_a_node(G_, top_percent_):
    delete_edge_tuple = []
    sorted_neighbor_weight_map = get_source_node_neighbor_weight_map(G_)
    final_num = math.floor(len(sorted_neighbor_weight_map) * top_percent_)
    for i in range(final_num):
        for node_ in ['CHN', 'HKG', 'MAC']:
            delete_edge_tuple.append((sorted_neighbor_weight_map[i][0], node_))
            delete_edge_tuple.append((node_, sorted_neighbor_weight_map[i][0]))
    return delete_edge_tuple


def get_G_eff_and_omega(G_air_, common_, remove_edge_tuples, Popu_, delta_t_):
    G_air_new = G_air_.copy()
    G_air_new.remove_edges_from(remove_edge_tuples)
    G_air_new.remove_edges_from(nx.selfloop_edges(G_air_new))
    F = np.array(nx.adjacency_matrix(G_air_new, common_, weight='weight').todense())
    F_update = (np.tril(F, -1) + np.transpose(np.triu(F, 1))) / 2
    F_update = F_update + F_update.T
    omega_ = F_update / 365 / Popu_[:, None] * delta_t_

    G_eff_ = nx.DiGraph()
    for (u, v, d) in G_air_new.edges(data=True):
        dis = - np.log(omega_[common_.index(u)][common_.index(v)])
        if dis > 0:
            G_eff_.add_edge(u, v, weight=dis)
        else:
            G_eff_.add_edge(u, v, weight=float('inf'))
    for node in G_air_new.nodes():
        if node not in G_eff_.nodes():
            G_eff_.add_node(node)
    return G_eff_, omega_


def get_eff_dis_list(common_, G_eff_, source_iso_code):
    eff_all_ = []
    for ta in common_:
        eff_all_.append(nx.shortest_path_length(G_eff_, source=source_iso_code, target=ta, weight='weight'))
    return eff_all_


def define_epidemic_paras(delta_t_):
    R0_ = 15  # basic reproduction number
    DI_ = 5.6 / delta_t_  # infectious period
    beta_ = R0_ / DI_
    lambda_ = beta_ - 1 / DI_
    gamma_ = 0.5772156649
    return beta_, DI_, lambda_, gamma_


def cross_minus_new(omega_, INFO_number):
    omega_info = omega_ * INFO_number[:, None]
    return np.sum(omega_info.T - omega_info, axis=1)


def numpy_sigma(x, eta=8):
    x_temp = np.power(x, eta)
    return np.true_divide(x_temp, 1 + x_temp)


def SIR_trans(N_, S_, I_, R_, omega_, beta_, infectious_period_, e=1e-10):
    SI_trans = S_ * I_ / N_ * beta_ * numpy_sigma(I_ / N_ / e)
    IR_trans = I_ / infectious_period_

    d_S = - SI_trans + cross_minus_new(omega_, S_)
    d_I = SI_trans - IR_trans + cross_minus_new(omega_, I_)
    d_R = IR_trans + cross_minus_new(omega_, R_)

    S_ += d_S
    I_ += d_I
    R_ += d_R

    S_ = np.minimum(Popu, np.maximum(np.zeros_like(Popu), S_))
    I_ = np.minimum(Popu, np.maximum(np.zeros_like(Popu), I_))
    R_ = np.minimum(Popu, np.maximum(np.zeros_like(Popu), R_))
    return S_, I_, R_,


# -------- GET parameters ----------------------------
delta_t = 0.05
seed_num = 100
node_num, common, Popu, G_air = init_data()
beta, DI, lambda_epi, gamma = define_epidemic_paras(delta_t)
# ----------------------------------------------------

# ------- travel restriction parameters --------------
source_index = common.index('CHN')  # 'CHN'
T = int(500 / delta_t)
top_percent_list = [0, 0.1, 0.5, 0.95]
all_neighbors_in_great_china = [t[0] for t in get_source_node_neighbor_weight_map(G_air)]
all_neighbors_in_great_china_index = [common.index(node) for node in all_neighbors_in_great_china]
all_neighbors_in_great_china_index += [common.index(node) for node in ['CHN', 'HKG', 'MAC']]
# ----------------------------------------------------

infected_time_all = {}
removed_nbrs = {}
effective_dis = {}
for top_percent in top_percent_list:
    # ----- travel restriction scenarios ----------------------------------
    # get new traffic network info
    removed_edges = get_top_edges_from_a_node(G_air, top_percent)
    G_eff, omega = get_G_eff_and_omega(G_air, common, removed_edges, Popu, delta_t)
    eff_all = get_eff_dis_list(common, G_eff, common[source_index])

    # get removed nbrs
    removed_nbrs_current = set()
    for edge in removed_edges:
        removed_nbrs_current.add(edge[0])
        removed_nbrs_current.add(edge[1])
    if len(removed_nbrs_current) > 0:
        removed_nbrs_current.remove(common[source_index])
        removed_nbrs_current.remove('HKG')
        removed_nbrs_current.remove('MAC')
    removed_nbrs[str(top_percent)] = list(removed_nbrs_current)

    # get effective distance from China
    effective_dis[str(top_percent)] = np.array(eff_all)[all_neighbors_in_great_china_index]
    # ---------------------------------------------------------------------

    # init
    I = np.zeros(node_num)
    I[source_index] = seed_num
    S = Popu - I
    R = np.zeros(node_num)

    # record
    infected_countries = set([])
    infected_countries.add(common[source_index])
    infected_time = np.ones(len(common)) * float('inf')
    infected_time[source_index] = 0

    # trans
    for t in range(1, T + 1):
        S, I, R = SIR_trans(Popu, S, I, R, omega, beta, DI)
        for xx in np.where(I >= 1)[0]:
            if common[xx] not in infected_countries:
                infected_time[xx] = t
                infected_countries.add(common[xx])
    infected_time_all[str(top_percent)] = np.array(infected_time)[all_neighbors_in_great_china_index] * delta_t

# ----------- saving results -------------------------------------
removed_nbrs_add_length = {p: removed_nbrs[p] + ['X'] * (len(removed_nbrs[str(top_percent_list[-1])])-
                                                         len(removed_nbrs[p]))
                           for p in removed_nbrs}

df = pd.DataFrame(removed_nbrs_add_length)
df.to_excel('results/final_results.xlsx', sheet_name='removed_nbrs', index=False)

df = pd.DataFrame(infected_time_all)
with pd.ExcelWriter('results/final_results.xlsx', engine="openpyxl", mode='a') as writer:
    df.to_excel(writer, sheet_name='infected_time', index=False)

df = pd.DataFrame(effective_dis)
with pd.ExcelWriter('results/final_results.xlsx', engine="openpyxl", mode='a') as writer:
    df.to_excel(writer, sheet_name='effective_dis', index=False)

df = pd.DataFrame([common[index] for index in all_neighbors_in_great_china_index])
with pd.ExcelWriter('results/final_results.xlsx', engine="openpyxl", mode='a') as writer:
    df.to_excel(writer, sheet_name='iso_code', index=False)

G_air_final = G_air.copy()
G_air_final.remove_edges_from(nx.selfloop_edges(G_air_final))  # remove self loop
china_circle = G_air_final.subgraph(all_neighbors_in_great_china + ['CHN', 'HKG', 'MAC'])
nodes_dict = {}
for node in ['CHN', 'HKG', 'MAC']:
    nodes_dict[node] = {'attr': 'china'}
for p in removed_nbrs:
    for node in removed_nbrs[p]:
        if node not in nodes_dict:
            nodes_dict[node] = {'attr': str(p)}
for node in all_neighbors_in_great_china:
    if node not in nodes_dict:
        nodes_dict[node] = {'attr': str(1)}
nx.set_node_attributes(china_circle, nodes_dict)
nx.write_gexf(china_circle, 'results/china_circle.gexf')

node_type = [0] * len(all_neighbors_in_great_china_index)
for node in nodes_dict:
    node_type[all_neighbors_in_great_china_index.index(common.index(node))] = nodes_dict[node]['attr']

df = pd.DataFrame(node_type)
with pd.ExcelWriter('results/final_results.xlsx', engine="openpyxl", mode='a') as writer:
    df.to_excel(writer, sheet_name='node_type', index=False)
