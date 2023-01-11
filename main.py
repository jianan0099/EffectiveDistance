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


def get_G_eff(G, common_, Popu_):
    G.remove_edges_from(nx.selfloop_edges(G))
    F = np.array(nx.adjacency_matrix(G, common_, weight='weight').todense())
    F_temp = (np.tril(F, -1) + np.transpose(np.triu(F, 1))) / 2
    F = F_temp + F_temp.T
    Fn = np.sum(F, axis=1)
    gamma_each_country = Fn/Popu_/365
    P = np.true_divide(F, np.transpose(np.tile(Fn, (len(F), 1))))
    G_eff = nx.DiGraph()
    for (u, v, d) in G.edges(data=True):
        F_inverse = G[v][u]['weight'] if (v, u) in G.edges() else 0
        if F_inverse > 0:
            G_eff.add_edge(u, v, weight=1 - math.log(F_inverse / Fn[common_.index(u)]))
        else:
            G_eff.add_edge(u, v, weight=float('inf'))
    for node in G.nodes():
        if node not in G_eff.nodes():
            G_eff.add_node(node)
    return G_eff, P, gamma_each_country


def get_big_phi(G):
    big_phi = 0
    for (u, v, d) in G.edges(data=True):
        big_phi += d['weight']
    big_phi = big_phi / 365
    return big_phi


def get_omega(popu_final):
    omega = sum(list(popu_final.values())) * 1000
    return omega


def get_eff_dis_list(common_, G_eff_, source_iso_code):
    eff_all_ = []
    for ta in common_:
        eff_all_.append(nx.shortest_path_length(G_eff_, source=source_iso_code, target=ta, weight='weight'))
    return eff_all_


def init_data():
    with open('final_data/common.json', 'r') as f:
        common_dict = json.load(f)
    common_ = common_dict['common']
    node_num_ = len(common_)
    with open('final_data/popu_without_1000.json', 'r') as f:
        popu_final = json.load(f)
    Popu_ = get_arr(popu_final, common_) * 1000
    G_air_ = nx.read_gexf('final_data/G_air_2019_complete_sample.gexf').subgraph(common_)
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


def get_G_eff_and_P(G_air_, common_, remove_edge_tuples,Popu_):
    G_air_new = G_air_.copy()
    G_air_new.remove_edges_from(remove_edge_tuples)
    G_eff_, P_, gamma_ = get_G_eff(G_air_new, common_, Popu_)
    return G_eff_, P_, gamma_


def define_epidemic_paras():
    R0 = 15  # https://www.medrxiv.org/content/10.1101/2022.12.25.22283940v2.full.pdf+html
    beta_ = 1 / 5
    alpha_ = R0 * beta_
    eta_ = 8
    e_ = 1e-10
    return beta_, alpha_, eta_, e_


def numpy_sigma(x, eta):
    x_temp = np.power(x, eta)
    return np.true_divide(x_temp, 1 + x_temp)


def cross_minus(arr, N):
    arr_temp1 = np.tile(arr, (N, 1))
    arr_temp2 = np.transpose(arr_temp1)
    return arr_temp1 - arr_temp2


def SIR_trans(N, P_, S_fractions_, I_fractions_, R_fractions_, Alpha_S, eta_, beta_, gamma_, e_):

    SI_sigma = np.multiply(numpy_sigma(I_fractions_ / e_, eta_), np.multiply(S_fractions_, I_fractions_))

    d_S_fractions = - Alpha_S * SI_sigma + gamma_ * np.sum(np.multiply(np.transpose(P_), cross_minus(S_fractions_, N)),
                                                           axis=1)
    d_I_fractions = Alpha_S * SI_sigma - beta_ * I_fractions_ + \
                    gamma_ * np.sum(np.multiply(np.transpose(P_), cross_minus(I_fractions_, N)), axis=1)
    d_R_fractions = beta_ * I_fractions_ + gamma_ * np.sum(np.multiply(np.transpose(P_), cross_minus(R_fractions_, N)),
                                                           axis=1)
    I_fractions_ += d_I_fractions
    S_fractions_ += d_S_fractions
    R_fractions_ += d_R_fractions

    I_fractions_ = np.minimum(np.ones(N), np.maximum(np.zeros(N), I_fractions_))
    S_fractions_ = np.minimum(np.ones(N), np.maximum(np.zeros(N), S_fractions_))
    R_fractions_ = np.minimum(np.ones(N), np.maximum(np.zeros(N), R_fractions_))

    return S_fractions_, I_fractions_, R_fractions_


def infer_disease_state(Population_arr, I_fractions_, N):
    disease_states_ = np.zeros(N)
    I_ = np.multiply(I_fractions_, Population_arr)

    for i in range(len(I_)):
        if I_[i] >= 1:
            disease_states_[i] = 1
        else:
            disease_states_[i] = 0
    return disease_states_


source_index = 125  # 'CHN'
T = 500
top_percent_list = [0, 0.1, 0.5, 0.95]
# basic info
node_num, common, Popu, G_air = init_data()
beta, alpha, eta, e = define_epidemic_paras()
all_neighbors_in_great_china = [t[0] for t in get_source_node_neighbor_weight_map(G_air)]
all_neighbors_in_great_china_index = [common.index(node) for node in all_neighbors_in_great_china]
all_neighbors_in_great_china_index += [common.index(node) for node in ['CHN', 'HKG', 'MAC']]
infected_time_all = {}
removed_nbrs = {}
effective_dis = {}
for top_percent in top_percent_list:
    # cal_eff
    removed_edges = get_top_edges_from_a_node(G_air, top_percent)
    G_eff, P, gamma = get_G_eff_and_P(G_air, common, removed_edges, Popu)
    eff_all = get_eff_dis_list(common, G_eff, common[source_index])
    removed_nbrs_current = set()
    for edge in removed_edges:
        removed_nbrs_current.add(edge[0])
        removed_nbrs_current.add(edge[1])
    if len(removed_nbrs_current) > 0:
        removed_nbrs_current.remove(common[source_index])
        removed_nbrs_current.remove('HKG')
        removed_nbrs_current.remove('MAC')
    removed_nbrs[str(top_percent)] = list(removed_nbrs_current)
    effective_dis[str(top_percent)] = np.array(eff_all)[all_neighbors_in_great_china_index]

    # init
    I_fractions = np.zeros(node_num)
    I_fractions[source_index] = 100 / Popu[source_index]
    S_fractions = 1 - I_fractions
    R_fractions = np.zeros(node_num)

    # record
    infected_countries = set([])
    infected_countries.add(common[source_index])
    infected_time = np.ones(len(common)) * float('inf')
    infected_time[source_index] = 0

    # trans
    for t in range(1, T + 1):
        S_fractions, I_fractions, R_fractions = SIR_trans(node_num, P, S_fractions, I_fractions, R_fractions,
                                             alpha, eta, beta, gamma, e)
        disease_states = infer_disease_state(Popu, I_fractions, node_num)
        for xx in np.where(disease_states > 0)[0]:
            if common[xx] not in infected_countries:
                infected_time[xx] = t
                infected_countries.add(common[xx])

    infected_time_all[str(top_percent)] = np.array(infected_time)[all_neighbors_in_great_china_index]
    plt.scatter(eff_all, infected_time, alpha=0.6)
    plt.show()

removed_nbrs_add_length = {p: removed_nbrs[p] + ['X'] * (len(removed_nbrs[str(top_percent_list[-1])])-len(removed_nbrs[p]))
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

detailed_countries = ['ITA', 'USA', 'JPN', 'GBR', 'KOR']
df = pd.DataFrame([all_neighbors_in_great_china_index.index(common.index(node)) for node in detailed_countries])
with pd.ExcelWriter('results/final_results.xlsx', engine="openpyxl", mode='a') as writer:
    df.to_excel(writer, sheet_name='detailed_countries_index', index=False)
df = pd.DataFrame(detailed_countries)
with pd.ExcelWriter('results/final_results.xlsx', engine="openpyxl", mode='a') as writer:
    df.to_excel(writer, sheet_name='detailed_countries', index=False)

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