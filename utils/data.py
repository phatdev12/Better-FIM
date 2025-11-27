import pandas as pd
import networkx as nx
import pickle

def load_data_from_pickle(pickle_file, attribute_name='group'):
    with open(pickle_file, 'rb') as f:
        g = pickle.load(f)
    
    G = g.to_undirected() if g.is_directed() else g
    common_attributes = ['group', 'color', 'region', 'ethnicity', 'age', 'gender', 'status', 'community', 'class']

    if attribute_name == 'auto':
        found_attr = None
        if len(G.nodes()) > 0:
            sample_node = list(G.nodes())[0]
            node_attrs = G.nodes[sample_node]
            
            # Thử các attribute phổ biến trước
            for attr in common_attributes:
                if attr in node_attrs:
                    found_attr = attr
                    break
            
            if not found_attr:
                for attr in node_attrs:
                    if attr not in ['pid', '_nx_name', 'label', 'name']:
                        found_attr = attr
                        break
        
        if not found_attr:
            raise ValueError("Không tìm thấy attribute nào trong graph nodes")
        
        attribute_name = found_attr
        print(f"Auto-detected attribute: {attribute_name}")
    
    # Trích xuất node_groups từ attribute đã chọn
    node_groups = {}
    for node in G.nodes():
        if attribute_name in G.nodes[node]:
            node_groups[node] = G.nodes[node][attribute_name]
            # Đảm bảo có attribute 'group' để tương thích
            if attribute_name != 'group':
                G.nodes[node]['group'] = G.nodes[node][attribute_name]
        else:
            # Nếu node không có attribute này, gán giá trị mặc định
            node_groups[node] = 0
            G.nodes[node]['group'] = 0
    
    return G, node_groups


def load_data(links_file, attr_file):
    edges_df = pd.read_csv(links_file, sep=r'\s+', header=None, names=['source', 'target'])
    G = nx.from_pandas_edgelist(edges_df, 'source', 'target') 
    attr_df = pd.read_csv(attr_file, sep=r'\s+', header=None, names=['node', 'group'])
    node_groups = attr_df.set_index('node')['group'].to_dict()

    for node_id, group_id in node_groups.items():
        G.add_node(node_id)
        G.nodes[node_id]['group'] = group_id
      
    return G, node_groups


def calculate_SN(G):
    return nx.pagerank(G)
