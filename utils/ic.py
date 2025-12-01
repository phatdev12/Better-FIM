import numpy as np
import random
import networkx as nx
from . import xp, to_numpy, GPU_AVAILABLE

def sample_live_icm(G, num_graphs, p=0.01):
    """
    Sample live edge graphs từ ICM (như CEA-FIM)
    Mỗi cạnh có xác suất p được activate
    
    Returns:
        List of live edge graphs (NetworkX graphs)
    """
    live_edge_graphs = []
    edges = list(G.edges())
    
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(G.nodes())
        
        # GPU-accelerated edge sampling
        if edges:
            success_gpu = xp.random.random(len(edges)) < p
            success = to_numpy(success_gpu) if GPU_AVAILABLE else success_gpu
            
            for i, (u, v) in enumerate(edges):
                if success[i]:
                    h.add_edge(u, v)
        
        live_edge_graphs.append(h)
    
    return live_edge_graphs


def run_IC_precomputed(S, live_graphs, target_nodes=None):
    """
    Chạy IC simulation trên pre-computed live graphs (FAST!)
    
    Args:
        S: seed set
        live_graphs: pre-computed live edge graphs
        target_nodes: optional target nodes to count
        
    Returns:
        average influence spread
    """
    total_spread = 0
    S_set = set(S)
    target_set = set(target_nodes) if target_nodes else None
    
    for h in live_graphs:
        # BFS trên live graph
        activated = S_set.copy()
        new_active = S_set.copy()
        
        while new_active:
            next_active = set()
            for node in new_active:
                if h.has_node(node):
                    for nbr in h.neighbors(node):
                        if nbr not in activated:
                            next_active.add(nbr)
                            activated.add(nbr)
            new_active = next_active
        
        if target_set:
            total_spread += len(activated.intersection(target_set))
        else:
            total_spread += len(activated)
    
    return total_spread / len(live_graphs)


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
    Thuật toán greedy với LAZY EVALUATION (như CEA-FIM)
    
    Args:
        G_sub: subgraph
        k: số lượng seed nodes
        p: xác suất lan truyền
        mc: số lần Monte Carlo
    
    Returns:
        influence spread của k nodes tốt nhất
    """
    import heapq
    
    if k > len(G_sub):
        return run_IC(G_sub, list(G_sub.nodes()), p, mc)
    
    S = []
    candidates = list(G_sub.nodes())
    
    # Pre-compute live graphs for faster evaluation
    live_graphs = sample_live_icm(G_sub, mc, p)
    
    # Initialize heap with upper bounds (negative for max-heap)
    upper_bounds = []
    for node in candidates:
        gain = run_IC_precomputed([node], live_graphs)
        heapq.heappush(upper_bounds, (-gain, node))
    
    current_influence = 0
    
    for _ in range(k):
        while True:
            neg_gain, node = heapq.heappop(upper_bounds)
            
            if node in S:
                continue
            
            # Recompute marginal gain
            new_influence = run_IC_precomputed(S + [node], live_graphs)
            marginal_gain = new_influence - current_influence
            
            # Check if this is still the best (lazy evaluation)
            if not upper_bounds or marginal_gain >= -upper_bounds[0][0] - 0.01:
                S.append(node)
                current_influence = new_influence
                break
            else:
                # Re-insert with updated gain
                heapq.heappush(upper_bounds, (-marginal_gain, node))
    
    return current_influence
