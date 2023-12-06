"""
Module for parsing python source code into an
Abstract Syntax Tree (AST) and converting the AST
into an adjacency matrix and node type array.

Example Usage:
    filepath = "gym_line_follower/line_follower_bot.py"
    adj_matrix, type_array = get_adjacency_matrix_and_type_array(filename, verbose=True)
"""

# Imports
import ast
import numpy as np
import scipy.linalg as linalg
import pdb 

# Node Type Constants
UNDEFINED_NODE = 0
HARDWARE_INPUT_NODE = -1
HARDWARE_OUTPUT_NODE = 1

def get_node_type(node):
    """
    Takes in a node object and evaluates whether it is a
    Hardware Input Node, Hardware Output Node, or Undefined Node.

    Inputs:
        node: Node object.

    Outputs:
        constant: Constant value representing the type of node.
    """
    if (isinstance(node, ast.Call) and 
        isinstance(node.func, ast.Attribute) and 
        isinstance(node.func.value, ast.Attribute)):

        # Check if node object represents a call to pb_client
        if node.func.value.attr == 'pb_client':
            #pdb.set_trace()
            if "get" in node.func.attr:
                return HARDWARE_INPUT_NODE
            elif "set" in node.func.attr:
                return HARDWARE_OUTPUT_NODE
                
    return UNDEFINED_NODE


def get_node_to_index_dict(ast_tree: ast):
    """
    Traverses the AST, assigns each node a unique ID, and
    creates a dictionary from node object to unique ID.

    Input:
        ast_tree: AST representation of source code.

    Output:
        node_to_index: Dictionary of node object to unique index.
    """
    node_to_index = {}
    index = 0
    for node in ast.walk(ast_tree):
        # Check if we've visited this node before
        if node not in node_to_index.keys():
            # If its the first time, add the node object as a key
            # in the dict with its value as the corresponding unique ID
            node_to_index[node] = index
            index += 1
    
    return node_to_index

def build_adj_and_type(ast_tree: ast, node_to_index: dict):
    """
    Traverses the AST to convert it into an adjacency matrix representation
    and a node type array representation.

    Inputs:
        ast_tree: AST representation of source code.
        node_to_index: Dictionary of node object to unique index.

    Outputs:
        adj_matrix: 2D Numpy adjacency matrix.
    """
    # Create zeros adjacency matrix with shape (num_node,num_nodes) and
    # zeros type array with length num_nodes
    num_nodes = len(node_to_index.keys())
    adj_matrix = np.zeros((num_nodes, num_nodes))
    type_array = np.zeros(num_nodes)
    lineno_array = np.zeros(num_nodes)


    # Loop over all nodes in AST
    for node in ast.walk(ast_tree):

        # Grad unique node ID
        node_index = node_to_index[node]

        # Evaluate if the node type is Hardware Input or Hardware Output
        node_type = get_node_type(node)
        type_array[node_index] = node_type
        if (isinstance(node, ast.Call) and 
        isinstance(node.func, ast.Attribute) and 
        isinstance(node.func.value, ast.Attribute)):
            lineno_array[node_index] = node.lineno
        else:
            lineno_array[node_index] = 0


        # Loop over all children nodes
        for child in ast.iter_child_nodes(node):

            # Grab unique child node ID
            child_index = node_to_index[child]

            # Update appropriate edges in adjacency matrix
            adj_matrix[node_index][child_index] = 1
            adj_matrix[child_index][node_index] = 1


    return adj_matrix, type_array, lineno_array


def get_adjacency_matrix_and_type_array(file_path: str, verbose: bool=False):
    """
    Opens file_path and converts the source code into an AST representation.
    Creates a dictionary that maps from each node object to a unique ID.
    Loops over the children of each node and creates an adjacency matrix that
    represents the edges connecting the nodes in the AST. 

    Inputs:
        file_path: Path to current source code file of interest.
        verbose: 

    Outputs:
        adj_matrix: 2D Numpy adjacency matrix.
        type_array: 1D Numpy type array.
    """
    # Parse source code into an AST
    with open(file_path, "r") as source:
        ast_tree = ast.parse(source.read(), filename="line_follower_bot")
    
    # Build dictionary that maps from node object to unique ID
    node_to_index = get_node_to_index_dict(ast_tree)

    # Build adjacency matrix representation and type array
    adj_matrix, type_array, lineo_array = build_adj_and_type(ast_tree, node_to_index)

    # Visualize AST, node_to_index dict, adjacecny matrix, and type array
    if verbose:
        print("\n --- AST DUMP ---")
        print(ast.dump(ast_tree))

        print("\n --- NODE_TO_INDEX DICTIONARY ---")
        for i in node_to_index.keys():
            print("ID & Node Object: ", node_to_index[i], "\t", i)

        print("\n --- ADJACECNY MATRIX ---")
        print(adj_matrix)
        
        print("\n --- TYPE ARRAY ---")
        print(type_array)
    
    return adj_matrix, type_array, node_to_index, lineo_array

def adjacency_matrix_to_edge_index(adj_matrix):
    edge_index = np.nonzero(adj_matrix)
    return edge_index

if __name__ == "__main__":
    """
    Example usage of this module. Call get_adjacecny_matrix_and_type_array on
    the file of interest.x
    """
    filename = "line_follower_bot.py"
    adj_matrix, type_array, node_to_index, lineo_array = get_adjacency_matrix_and_type_array(filename, verbose=True)
    for i in range(len(type_array)):
        if type_array[i] != 0:
            print(lineo_array[i])
    pdb.set_trace()
