import numpy as np
import random
from . import xp, to_numpy, GPU_AVAILABLE

def run_IC(G, S, p=0.01, mc=50, target_nodes=None):
    """
    Chạy mô phỏng Independent Cascade để tính influence spread
    
    Args:
        G: đồ thị NetworkX
        S: tập seed nodes
        p: xác suất lan truyền
        mc: số lần Monte Carlo simulation
        target_nodes: nếu chỉ định, chỉ đếm influence trong tập này
    
    Returns:
        average influence spread
    """
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
                    if not neighbors:
                        continue
                    
                    # GPU-accelerated Bernoulli sampling
                    success_gpu = xp.random.random(len(neighbors)) < p
                    success = to_numpy(success_gpu) if GPU_AVAILABLE else success_gpu
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
    """
    Thuật toán greedy (không lazy evaluation)
    
    Args:
        G_sub: subgraph
        k: số lượng seed nodes
        p: xác suất lan truyền
        mc: số lần Monte Carlo
    
    Returns:
        influence spread của k nodes tốt nhất
    """
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
            if node in S:
                continue
            
            spread = run_IC(G_sub, S + [node], p, mc=10)
            
            if spread > max_gain:
                max_gain = spread
                best_node = node
        
        if best_node:
            S.append(best_node)
    
    return run_IC(G_sub, S, p, mc=mc)
