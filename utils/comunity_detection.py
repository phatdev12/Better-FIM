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
        if count == 0:
            u_j[g_id] = 0
        else:
            u_j[g_id] = 1.0
    
    SC = {}
    for c_id, nodes in communities.items():
        size = len(nodes)
        
        # Thu thập các attributes (groups) trong community
        comm_attrs = set()
        for n in nodes:
            if G.has_node(n) and 'group' in G.nodes[n]:
                comm_attrs.add(G.nodes[n]['group'])
                
        # Tính tổng urgency của các attributes trong community
        sum_urgency = sum([u_j.get(attr, 0) for attr in comm_attrs])
        SC[c_id] = size * sum_urgency
        
    return SC


def community_based_selection(G, k, communities, SN_scores, SC_scores):
    seed_set = set()
    
    comm_ids = list(SC_scores.keys())
    total_SC = sum(SC_scores.values())
    
    # Tính xác suất chọn community
    if total_SC == 0: 
        probs_C = [1.0/len(comm_ids)] * len(comm_ids)
    else: 
        probs_C = [SC_scores[c]/total_SC for c in comm_ids]
    
    for _ in range(k):
        # Chọn community
        chosen_comm_id = np.random.choice(comm_ids, p=probs_C)
        chosen_nodes = communities[chosen_comm_id]
        
        # Lọc các nodes chưa được chọn
        candidates = [n for n in chosen_nodes if n not in seed_set]
        if not candidates:
            # Nếu community đã hết, chọn từ toàn đồ thị
            remaining = list(set(G.nodes()) - seed_set)
            if remaining:
                seed_set.add(np.random.choice(remaining))
            continue
            
        # Chọn node theo SN score
        total_SN = sum([SN_scores.get(n, 0) for n in candidates])
        if total_SN == 0: 
            probs_N = [1.0/len(candidates)] * len(candidates)
        else: 
            probs_N = [SN_scores.get(n, 0)/total_SN for n in candidates]
        
        chosen_node = np.random.choice(candidates, p=probs_N)
        seed_set.add(chosen_node)
        
    return list(seed_set)
