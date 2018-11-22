import pandas as pd
import networkx as nx
from texttable import Texttable

def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(args.edge_path).values.tolist())
    return graph

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def loss_printer(losses):
    """
    Printing the losses for each iteration.
    :param losses: List of losses in each iteration.
    """
    t = Texttable() 
    t.add_rows([["Iteration","Reconstrcution Loss I.","Reconstruction Loss II.","Regularization Loss"]] +  losses)
    print(t.draw())
