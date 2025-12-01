from . import ic

def _seed_key(S):
    try:
        return tuple(sorted(S))
    except Exception:
        return tuple(S)

def calculate_MF_DCV(G, S, groups, ideal_influences, p=0.01, mc=50, cache=None):
    """
    Tính MF (Min Fraction) và DCV (Deviation from Coverage Violation)
    
    MF = min(actual_influence / group_size) - coverage tối thiểu cho các nhóm
    DCV = avg(max(0, (ideal - actual) / ideal)) - độ lệch trung bình so với ideal
    """
    unique_groups = list(groups.keys())
    
    min_fraction = float('inf')
    total_violation = 0
    
    seed_key = _seed_key(S)
    for g_id in unique_groups:
        group_nodes = groups[g_id]
        if not group_nodes: 
            continue
        if cache is not None:
            k = (seed_key, g_id)
            if k in cache:
                actual_inf = cache[k]
            else:
                actual_inf = ic.run_IC(G, S, p, mc, target_nodes=group_nodes)
                cache[k] = actual_inf
        else:
            actual_inf = ic.run_IC(G, S, p, mc, target_nodes=group_nodes)
        
        group_size = len(group_nodes)
        fraction = actual_inf / group_size if group_size > 0 else 0
        
        if fraction < min_fraction:
            min_fraction = fraction
        
        ideal = ideal_influences.get(g_id, 0)
        if ideal == 0:
            ideal = 0.0001  # Tránh chia cho 0
        
        violation = max(0, (ideal - actual_inf) / ideal)
        total_violation += violation
        
    MF = min_fraction
    DCV = total_violation / len(unique_groups) if len(unique_groups) > 0 else 0
    
    return MF, DCV
