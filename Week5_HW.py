# WEEK 5 PROBLEM SET - HOMEWORK

# HW1. Dictionary: Write two functions:
# count_degrees(G): which sums up the degrees of all vertices in the graph. 
#                   The degree of a vertex is defined as the number of edges connected to a Vertex. 
#                   The argument G is a dictionary that represents the graph.
# count_edges(G): which counts the number of edges in the graph. 
#                 An edge is defined as a connection between two vertices.
#                 The argument G is a dictionary.

def count_degrees(G):
    result = 0
    for key in G:
        result += len(G[key])
    return result

def count_edges(G):
    total_edges = sum([len(neighbours) for neighbours in G.values()])
    return total_edges // 2

g1 = {"A": ["B", "E"], 
        "B": ["A", "C"],
        "C": ["B", "D", "E"],
        "D": ["C"],
        "E": ["A", "C"]}

d = count_degrees(g1)
e = count_edges(g1)
print(d)
print (e)