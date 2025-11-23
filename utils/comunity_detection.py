import leidenalg
import igraph as ig 
import numpy as np

def get_community_structure(G):
    G_ig = ig.Graph.from_networkx(G)
    
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    communities = {}
    
    for idx, comm_id in enumerate(partition.membership):
        if comm_id not in communities: 
            communities[comm_id] = []
        original_node_id = G_ig.vs[idx]['_nx_name']
        communities[comm_id].append(original_node_id)
        
    return communities, partition

def calculate_SC(communities, G, selected_nodes_attrs, A_j_counts):
    u_j = {}
    for g_id, count in A_j_counts.items():
        if count == 0: u_j[g_id] = 0
        else:
            u_j[g_id] = 1.0 
    
    SC = {}
    for c_id, nodes in communities.items():
        size = len(nodes)
        comm_attrs = set()
        for n in nodes:
            if G.has_node(n):
                comm_attrs.add(G.nodes[n]['group'])
                
        sum_urgency = sum([u_j[attr] for attr in comm_attrs if attr in u_j])
        SC[c_id] = size * sum_urgency
        
    return SC

def community_based_selection(G, k, communities, SN_scores, SC_scores):
    seed_set = set()
    
    comm_ids = list(SC_scores.keys())
    total_SC = sum(SC_scores.values())
    
    if total_SC == 0: 
        probs_C = [1.0/len(comm_ids)] * len(comm_ids)
    else: 
        probs_C = [SC_scores[c]/total_SC for c in comm_ids]
    
    for _ in range(k):
        chosen_comm_id = np.random.choice(comm_ids, p=probs_C)
        chosen_nodes = communities[chosen_comm_id]
        
        candidates = [n for n in chosen_nodes if n not in seed_set]
        if not candidates:
            remaining = list(set(G.nodes()) - seed_set)
            if remaining: seed_set.add(np.random.choice(remaining))
            continue
            
        total_SN = sum([SN_scores[n] for n in candidates])
        if total_SN == 0: 
            probs_N = [1.0/len(candidates)] * len(candidates)
        else: 
            probs_N = [SN_scores[n]/total_SN for n in candidates]
        
        chosen_node = np.random.choice(candidates, p=probs_N)
        seed_set.add(chosen_node)
        
    return list(seed_set)
