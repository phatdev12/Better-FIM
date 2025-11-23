import numpy as np
import random

def run_IC(G, S, p=0.01, mc=50, target_nodes=None):
    spread = 0
    S_set = set(S)
    target_set = set(target_nodes) if target_nodes is not None else None
    
    for _ in range(mc):
        new_active = S_set.copy()
        current_active = S_set.copy()
        
        while new_active:
            new_ones = set()
            for node in new_active:
                if G.has_node(node):
                    neighbors = list(G.neighbors(node))
                    if not neighbors: continue
                    
                    success = np.random.random(len(neighbors)) < p
                    for i, nbr in enumerate(neighbors):
                        if success[i] and nbr not in current_active:
                            new_ones.add(nbr)
                            current_active.add(nbr)
            new_active = new_ones
            
        if target_set:
            spread += len(current_active.intersection(target_set))
        else:
            spread += len(current_active)
            
    return spread / mc

def greedy_max_influence(G_sub, k, p=0.01, mc=20):
    S = []

    if k > len(G_sub):
        return run_IC(G_sub, list(G_sub.nodes()), p, mc)

    for _ in range(k):
        best_node = None
        max_gain = -1
        candidates = list(G_sub.nodes()) 
        if len(candidates) > 100: 
             candidates = random.sample(candidates, 100)

        for node in candidates:
            if node in S: continue
            spread = run_IC(G_sub, S + [node], p, mc=10)
            if spread > max_gain:
                max_gain = spread
                best_node = node
        
        if best_node:
            S.append(best_node)
            
    return run_IC(G_sub, S, p, mc=mc)
