from utils import data, comunity_detection, mf_dcv, ic, fitness
from utils import xp, to_numpy, GPU_AVAILABLE
from multiprocessing import Pool, cpu_count
import networkx as nx
import numpy as np
import random
import math
import copy

K_SEEDS = 40
POP_SIZE = 10
MAX_GEN = 150
P_CROSSOVER = 0.6
P_MUTATION = 0.1
LAMBDA_VAL = 0.5
PROPAGATION_PROB = 0.01
MC_SIMULATIONS = 1000  # Số lần mô phỏng Monte Carlo

def evuluate(individual, G, groups, ideal_influences, cache):
    mf, dcv = mf_dcv.calculate_MF_DCV(
        G, individual, groups, ideal_influences,
        p=PROPAGATION_PROB, mc=MC_SIMULATIONS, cache=cache
    )
    fit = fitness.fitness_F(mf, dcv, LAMBDA_VAL)
    return individual, mf, dcv, fit

def init_betterFIM_data(links_file, attr_file=None, attribute_name='color'):
    try:
        if links_file.endswith('.pickle') or links_file.endswith('.pkl'):
            G, node_groups_map = data.load_data_from_pickle(links_file, attribute_name)
        else:
            G, node_groups_map = data.load_data(links_file, attr_file)
    except FileNotFoundError:
        print(f"Error: Không tìm thấy file data. Hãy đảm bảo {links_file} tồn tại.")
        return None
    groups = {}
    for n, g in node_groups_map.items():
        if g not in groups: groups[g] = []
        groups[g].append(n)
    
    print(f"Nodes: {len(G)}, Edges: {G.number_of_edges()}")
    print(f"Groups: {list(groups.keys())}")
    
    SN_scores = nx.pagerank(G)
    communities, _ = comunity_detection.get_community_structure(G)
    A_j_counts = {g: len(nodes) for g, nodes in groups.items()}
    SC_scores = comunity_detection.calculate_SC(communities, G, None, A_j_counts)
    N = len(G)
    ideal_influences = {}
    for g_id, nodes in groups.items():
        k_i = math.ceil(K_SEEDS * len(nodes) / N)
        subgraph = G.subgraph(nodes)
        ideal = ic.greedy_max_influence(subgraph, k_i, p=PROPAGATION_PROB, mc=30)
        ideal_influences[g_id] = ideal
    return {
        'G': G,
        'groups': groups,
        'SN_scores': SN_scores,
        'communities': communities,
        'SC_scores': SC_scores,
        'ideal_influences': ideal_influences
    }

def betterFIM_main(data_obj):
    G = data_obj['G']
    groups = data_obj['groups']
    SN_scores = data_obj['SN_scores']
    communities = data_obj['communities']
    SC_scores = data_obj['SC_scores']
    ideal_influences = data_obj['ideal_influences']
    
    population = []
    
    for _ in range(POP_SIZE):
        ind = comunity_detection.community_based_selection(G, K_SEEDS, communities, SN_scores, SC_scores)
        population.append(ind)
        
        community_counter = {cid: 0 for cid in communities.keys()}
        community_scores = {cid: SC_scores.get(cid, 0) for cid in communities.keys()}
        community_selected = {cid: 0 for cid in communities.keys()}
        
        fairness_random_solution = []
        for _ in range(0, K_SEEDS):
            total_score = sum(community_scores.values())
            if total_score == 0:
                selected_comm_id = random.choice(list(communities.keys()))
            else:
                random_value = random.random() * total_score
                cumulative = 0
                for cid, score in community_scores.items():
                    cumulative += score
                    if random_value <= cumulative:
                        selected_comm_id = cid
                        break
            community_counter[selected_comm_id] += 1
        
            if not community_selected[selected_comm_id]:
                community_scores[selected_comm_id] = SC_scores.get(selected_comm_id, 0)
                community_selected[selected_comm_id] = 1
        for cid, count in community_counter.items():
            if count > 0:
                nodes_in_comm = list(communities[cid])
                sorted_nodes = sorted(nodes_in_comm, key=lambda x: SN_scores.get(x, 0), reverse=True)
                selected_nodes = sorted_nodes[:count]
                fairness_random_solution.extend(selected_nodes)
        
        while len(fairness_random_solution) < K_SEEDS:
            all_nodes = sorted(G.nodes(), key=lambda x: SN_scores.get(x, 0), reverse=True)
            for node in all_nodes: 
                if node not in fairness_random_solution:
                    fairness_random_solution.append(node)
                    break
        fairness_random_solution = fairness_random_solution[:K_SEEDS]
        population.append(fairness_random_solution)
        
        all_nodes = list(G.nodes())
        if all_nodes:
            weights_gpu = xp.array([SN_scores.get(n, 0) + 1e-8 for n in all_nodes], dtype=float)
            if float(xp.sum(weights_gpu)) == 0:
                random_weighted_solution = random.sample(all_nodes, min(K_SEEDS, len(all_nodes)))
            else:
                probs_gpu = weights_gpu / xp.sum(weights_gpu)
                k_pick = min(K_SEEDS, len(all_nodes))
                # Use GPU sampling when available
                if GPU_AVAILABLE:
                    # Build gp arrays and sample indices on GPU, then map back to nodes
                    probs_gpu = probs_gpu / xp.sum(probs_gpu)
                    probs_cpu = to_numpy(probs_gpu)
                    idxs = np.random.choice(len(all_nodes), size=k_pick, replace=False, p=probs_cpu)
                    random_weighted_solution = [all_nodes[int(i)] for i in idxs]
                else:
                    probs = to_numpy(probs_gpu)
                    random_weighted_solution = list(np.random.choice(all_nodes, size=k_pick, replace=False, p=probs))
            population.append(random_weighted_solution)

    best_S = None
    best_Fit = -999
    best_metrics = (0, 0)

    for gen in range(MAX_GEN):
        influence_cache = {}
        results = []
        for ind in population:
            result = evuluate(ind, G, groups, ideal_influences, influence_cache)
            results.append(result)

        fitnesses = []
        for ind, mf, dcv, fit in results:
            fitnesses.append(fit)

            if fit > best_Fit:
                best_Fit = fit
                best_S = ind
                best_metrics = (mf, dcv)
        
        fitnesses_gpu = xp.array(fitnesses)
        sorted_idx_gpu = xp.argsort(fitnesses_gpu)[::-1]
        sorted_idx = to_numpy(sorted_idx_gpu)
        # Selection: keep top N (elites)
        population = [population[i] for i in sorted_idx[:POP_SIZE]]

        # Inject ~N/3 random solutions to avoid premature convergence
        def make_random_solution():
            sol = []
            comm_keys_local = list(communities.keys())
            while len(sol) < K_SEEDS:
                # Step 1: random community and random node in it
                c_id = random.choice(comm_keys_local)
                c_nodes = list(communities[c_id])
                if not c_nodes:
                    continue
                v = random.choice(c_nodes)
                # Step 2: add if not present
                if v not in sol:
                    sol.append(v)
                # Step 3: stop when |S| == k (handled by while condition)
            return sol

        inject_count = max(1, POP_SIZE // 3)
        for _ in range(inject_count):
            population.append(make_random_solution())

        new_pop = []
        # Keep top-2 strong elites explicitly
        new_pop.extend(population[:2])
        
        while len(new_pop) < POP_SIZE:
            idx1 = np.random.randint(0, len(population))
            idx2 = np.random.randint(0, len(population))
            p1 = population[idx1]
            p2 = population[idx2]
            
            if np.random.random() < P_CROSSOVER:
                combined = list(set(p1) | set(p2))
                combined.sort(key=lambda x: SN_scores.get(x, 0), reverse=True)
                child = combined[:K_SEEDS]
            else:
                child = p1[:]
            
            if np.random.random() < P_MUTATION and len(child) > 0:
                idx_remove = np.random.randint(0, len(child))
                removed_node = child.pop(idx_remove)
                group_coverage = {g_id: 0 for g_id in groups.keys()}
                for node in child:
                    for g_id, g_nodes in groups.items():
                        if node in g_nodes:
                            group_coverage[g_id] += 1
                            break
                comm_weights = {}
                for cid, comm_nodes in communities.items():
                    weight = 0
                    for g_id, g_nodes in groups.items():
                        overlap = len(set(comm_nodes) & set(g_nodes))
                        if overlap > 0:
                            ideal_count = max(1, int(K_SEEDS * len(g_nodes) / len(G)))
                            deficit = max(0, ideal_count - group_coverage[g_id])
                            weight += deficit * overlap
                    comm_weights[cid] = weight + 1
                comm_keys = list(communities.keys())
                weights_gpu = xp.array([comm_weights[cid] for cid in comm_keys], dtype=float)
                weights_gpu = weights_gpu / xp.sum(weights_gpu)
                if GPU_AVAILABLE:
                    probs_gpu = weights_gpu / xp.sum(weights_gpu)
                    probs_cpu = to_numpy(probs_gpu)
                    idx = np.random.choice(len(comm_keys), p=probs_cpu)
                    comm_id = comm_keys[int(idx)]
                else:
                    weights_arr = to_numpy(weights_gpu)
                    comm_id = np.random.choice(comm_keys, p=weights_arr)
                candidates = list(communities[comm_id])
                if candidates:
                    candidates_not_in = [c for c in candidates if c not in child]
                    if candidates_not_in:
                        sn_vals_gpu = xp.array([SN_scores.get(c, 0) for c in candidates_not_in], dtype=float)
                        if float(xp.sum(sn_vals_gpu)) > 0:
                            if GPU_AVAILABLE:
                                probs_gpu = sn_vals_gpu / xp.sum(sn_vals_gpu)
                                probs_cpu = to_numpy(probs_gpu)
                                idx = np.random.choice(len(candidates_not_in), p=probs_cpu)
                                cand = candidates_not_in[int(idx)]
                            else:
                                probs_gpu = sn_vals_gpu / xp.sum(sn_vals_gpu)
                                probs = to_numpy(probs_gpu)
                                cand = np.random.choice(candidates_not_in, p=probs)
                        else:
                            if GPU_AVAILABLE:
                                idx = np.random.choice(len(candidates_not_in))
                                cand = candidates_not_in[int(idx)]
                            else:
                                cand = np.random.choice(candidates_not_in)
                        child.append(cand)
                    elif len(p1) > idx_remove:
                        child.append(p1[idx_remove])
                    else:
                        child.append(removed_node)
            while len(child) < K_SEEDS:
                possible = list(set(G.nodes()) - set(child))
                if not possible: break
                child.append(np.random.choice(possible))
            child = child[:K_SEEDS]
            new_pop.append(child)
        population = new_pop
    return best_Fit, best_metrics, best_S

def betterFIM(*args, **kwargs):
    """
    Wrapper: Cho phép gọi betterFIM(data_obj) hoặc betterFIM(links_file, attr_file, ...)
    """
    if len(args) == 1:
        # Đã truyền data_obj
        return betterFIM_main(args[0])
    elif len(args) >= 2:
        # Truyền links_file, attr_file, ...
        links_file = args[0]
        attr_file = args[1]
        attribute_name = args[2] if len(args) > 2 else 'color'
        data_obj = init_betterFIM_data(links_file, attr_file, attribute_name)
        return betterFIM_main(data_obj)
    else:
        raise TypeError("betterFIM requires either (data_obj) or (links_file, attr_file, [attribute_name])")