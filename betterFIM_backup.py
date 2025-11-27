from utils import data, comunity_detection, mf_dcv, ic, fitness
from multiprocessing import Pool, cpu_count
import numpy as np
import random
import math

K_SEEDS = 40
POP_SIZE = 10
MAX_GEN = 150
P_CROSSOVER = 0.6
P_MUTATION = 0.1
LAMBDA_VAL = 0.5
PROPAGATION_PROB = 0.01
MC_SIMULATIONS = 1000

def evuluate(individual, G, groups, ideal_influences):
    mf, dcv = mf_dcv.calculate_MF_DCV(G, individual, groups, ideal_influences, p=PROPAGATION_PROB, mc=MC_SIMULATIONS)
    fit = fitness.fitness_F(mf, dcv, LAMBDA_VAL)
    return individual, mf, dcv, fit

def betterFIM(links_file, attr_file=None, attribute_name='auto'):
    try:
        if links_file.endswith('.pickle') or links_file.endswith('.pkl'):
            G, node_groups_map = data.load_data_from_pickle(links_file, attribute_name)
        else:
            G, node_groups_map = data.load_data(links_file, attr_file)
    except FileNotFoundError:
        print(f"Error: Không tìm thấy file data. Hãy đảm bảo {links_file} tồn tại.")
        return

    groups = {}
    for n, g in node_groups_map.items():
        if g not in groups: groups[g] = []
        groups[g].append(n)
    
    print(f"Nodes: {len(G)}, Edges: {G.number_of_edges()}")
    print(f"Groups: {list(groups.keys())}") 

    ideal_influences = {}
    N = len(G)
    for g_id, nodes in groups.items():
        k_i = math.ceil(K_SEEDS * len(nodes) / N)
        subgraph = G.subgraph(nodes)
        ideal = ic.greedy_max_influence(subgraph, k_i, p=PROPAGATION_PROB, mc=30)
        ideal_influences[g_id] = ideal
        
    SN_scores = data.calculate_SN(G)
    
    communities, _ = comunity_detection.get_community_structure(G)
    
    A_j_counts = {g: len(nodes) for g, nodes in groups.items()}
    SC_scores = comunity_detection.calculate_SC(communities, G, None, A_j_counts)

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
            weights = np.array([SN_scores.get(n, 0) + 1e-8 for n in all_nodes])
            if weights.sum() == 0:
                random_weighted_solution = random.sample(all_nodes, min(K_SEEDS, len(all_nodes)))
            else:
                probs = weights / weights.sum()
                k_pick = min(K_SEEDS, len(all_nodes))
                random_weighted_solution = list(np.random.choice(all_nodes, size=k_pick, replace=False, p=probs))
            population.append(random_weighted_solution)

    best_S = None
    best_Fit = -999
    best_metrics = (0, 0)

    for gen in range(MAX_GEN):
        fitnesses = []
        
        # Evaluate
        for ind in population:
            mf, dcv = mf_dcv.calculate_MF_DCV(G, ind, groups, ideal_influences, p=PROPAGATION_PROB, mc=MC_SIMULATIONS)
            fit = fitness.fitness_F(mf, dcv, LAMBDA_VAL)
            fitnesses.append(fit)
            
            if fit > best_Fit:
                best_Fit = fit
                best_S = ind
                best_metrics = (mf, dcv)
        
        # Selection
        sorted_idx = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_idx[:POP_SIZE]]

        # print(f"Gen {gen+1}: Best Fit={best_Fit:.4f} | MF={best_metrics[0]:.4f}, DCV={best_metrics[1]:.4f}")

        # Crossover & Mutation
        new_pop = []
        new_pop.extend(population[:2]) # Elitism
        
        while len(new_pop) < POP_SIZE:
            idx1 = np.random.randint(0, len(population))
            idx2 = np.random.randint(0, len(population))
            p1 = population[idx1]
            p2 = population[idx2]
            
            # Crossover
            if np.random.random() < P_CROSSOVER:
                combined = list(set(p1) | set(p2))
                # Sort theo SN score để lấy top node
                combined.sort(key=lambda x: SN_scores.get(x, 0), reverse=True)
                child = combined[:K_SEEDS]
            else:
                child = p1[:]
            
            # Mutation
            if np.random.random() < P_MUTATION and len(child) > 0:
                idx_remove = np.random.randint(0, len(child))
                child.pop(idx_remove)
                
                comm_keys = list(communities.keys())
                if comm_keys:
                    comm_id = np.random.choice(comm_keys)
                    candidates = communities[comm_id]
                    if candidates:
                        cand = np.random.choice(candidates)
                        if cand not in child: child.append(cand)
                        elif len(p1) > idx_remove: child.append(p1[idx_remove])
            
            # Fill if missing
            while len(child) < K_SEEDS:
                possible = list(set(G.nodes()) - set(child))
                if not possible: break
                child.append(np.random.choice(possible))
                
            
            # Trim if excess
            child = child[:K_SEEDS]
            
            new_pop.append(child)
            
        population = new_pop
    
    
    return best_Fit, best_metrics, best_S