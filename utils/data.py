import pandas as pd
import networkx as nx

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
