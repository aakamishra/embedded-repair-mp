"""
Module for parsing python source code into an
Abstract Syntax Tree (AST).
"""

# Imports
import ast

# Constants
verbose = True
filename = "gym_line_follower/line_follower_bot.py"


def locate_pb_client_calls(ast_tree):
    """
    Traverse the AST to locate all calls to 'self.pb_client'.
    """
    
    # Containers
    hardware_input_nodes = []
    hardware_output_nodes = []
    undefined_nodes = []
    num_calls = 0

    # Traverse AST
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            if node.func.value.attr == 'pb_client':

                # Capture important function call information
                node_dict = {'pb_client_obj_str': ast.unparse(node.func.value),
                                'pb_client_obj_ast': node.func.value,
                                'pb_client func call': node.func.attr,
                                'args':[ast.unparse(j) for j in node.args],
                                'kwargs':{j.arg:ast.unparse(j.value) for j in node.keywords}}

                # Identify type of pb_client call
                if "get" in node.func.attr:
                    hardware_input_nodes.append(node_dict)
                elif "set" in node.func.attr:
                    hardware_output_nodes.append(node_dict)
                else:
                    undefined_nodes.append(node_dict)

                num_calls += 1

    return hardware_input_nodes, hardware_output_nodes, undefined_nodes, num_calls

if __name__ == "__main__":
    # Parse line_follower_bot.py into an AST
    with open(filename, "r") as source:
        ast_tree = ast.parse(source.read(), filename="line_follower_bot")

    # Visualize AST
    if verbose:
        print(ast.dump(ast_tree, indent=4))

    hardware_input_nodes, hardware_output_nodes, undefined_nodes, num_calls = locate_pb_client_calls(ast_tree)

    print("\n-- AST Results --")
    print("\nVariable self.pb_client is called ", num_calls, " times in line_follower_bot.py")
    print("\nNumber of detected Hardware Input Nodes: ", len(hardware_input_nodes))
    for i in hardware_input_nodes:
        print(i, "\n")
    print("\nNumber of detected Hardware Output Nodes: ", len(hardware_output_nodes))
    for j in hardware_output_nodes:
        print(j, "\n")
    print("\nNumber of Undefined Nodes: ", len(undefined_nodes))
    for k in undefined_nodes:
        print(k, "\n")