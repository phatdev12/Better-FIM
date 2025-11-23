from . import ic

def calculate_MF_DCV(G, S, groups, ideal_influences, p=0.01, mc=50):
    unique_groups = list(groups.keys())
    
    min_ratio = float('inf')
    total_dcv = 0
    
    for g_id in unique_groups:
        group_nodes = groups[g_id]
        if not group_nodes: continue

        actual_inf = ic.run_IC(G, S, p, mc, target_nodes=group_nodes)
        
        ratio = actual_inf / len(group_nodes)
        if ratio < min_ratio: min_ratio = ratio
        
        ideal = ideal_influences.get(g_id, 0)
        if ideal == 0: ideal = 0.0001
        
        val = (ideal - actual_inf) / ideal
        total_dcv += max(val, 0)
        
    MF = min_ratio
    DCV = total_dcv / len(unique_groups)
    
    return MF, DCV
