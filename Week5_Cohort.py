# WEEK 5 PROBLEM SET - COHORT

# CS1. Dictionary: Implement a Graph using a Dictionary where the keys are the 
#                  Vertices in the Graph and the values (in the the key-value pair) 
#                  correspond to an Array containing the neighbouring Vertices. 
#                   For example, let's represent the following graph:

#                   A -> B
#                   A -> C
#                   B -> C
#                   B -> D
#                   C -> D
#                   D -> C
#                   E -> F
#                   F -> C
# Create a Dictionary to represent the graph above.

graph = {
    'A': ['B', 'C'], 
    'B': ['C', 'D'],
    'C': ['D'],
    'D': ['C'],
    'E': ['F'],
    'F': ['C']
}

# test case 
print(graph)

# Write a function get_neighbours(graph, vert) 
# which returns a list of all neighbours of the requested Vertex vert in the graph. 
# Return None if the Vertex is not in the graph.

def get_neighbours(graph, vert):
    return graph.get(vert)

# test case 
assert get_neighbours(graph, "B") == ["C", "D"]
assert get_neighbours(graph, "A") == ["B", "C"]
assert get_neighbours(graph, "F") == ["C"]
assert get_neighbours(graph, "Z") == None

# Write a function get_source(graph, vert) 
# which returns a list of all source Vertices pointing to vert in the graph. 
# For example, Vertex "C" has the following source Vertices: ["A", "B", "D", "F"]. 
# Return a None object if there are none.

def get_source(graph, vert):
    source_vertices = []

    for vertex, neighbours in graph.items():
        if vert in neighbours:
            source_vertices.append(vertex)
    
    return source_vertices if source_vertices != [] else None

# test case 
assert sorted(get_source(graph, "C")) == ["A", "B", "D", "F"]
assert sorted(get_source(graph, "D")) == ["B", "C"]
assert sorted(get_source(graph, "F")) == ["E"]
assert get_source(graph, "Z") == None

# CS2. Create a class Vertex to represent a vertex in a graph. 
# The class Vertex has the following attributes:
# (1) id_: to identify each vertex. This is of String data type.
# (2) neighbours: which is a Dictionary where the keys are the neighbouring Vertex object instances 
#     that are connected to the current Vertex and the values are the weights of the edge between 
#     the current Vertex and the neighbouring vertices.

# The class should also have the following methods:
# (1) __init__(self, id_): 
#     which is used to initialized the attribute id_. 
#     By default, id_ is set to an empty String . 
#     The attribute neighbours is always set to an empty dictionary.
# (2) add_neighbour(self, nbr_vertex, weight): 
#     which adds a neighbouring Vertex to the current Vertex. 
#     The second argument provides the weight of the edge between the 
#     current Vertex and the newly added neighbouring Vertex. By default, weight is 0.
# (3) get_neigbours(self): 
#     which returns all the Vertices connected to the current Vertex as a list. 
#     The elements of the output list are of Vertex object instances.
# (4) get_weight(self, neighbour): 
#     which returns the weight of the requested neighbour. 
#     It should return None if the requested neighbour is not found.
# (5) __eq__(self, other): 
#     which returns true if the id of the current vertex object is the same as the other vertex's id.
# (6) __lt__(self, other): 
#     which returns true if the id of the current vertex object is less than the other vertex's id.
# (7) __hash__(self): 
#     which calls the hash() function on id_ and returns it. 
#     This allows the object to be a dictionary key. This is provided for you.
# (8) __str__(self): 
#     This method should return the id of the current vertex 
#     and a list of id_s of the neighbouring vertices, like Vertex 2 is connected to: 3, 4, 5 .

class Vertex:
    
    def __init__(self, id_=""):
        self.id_ = id_
        self.neighbours = {}

    def add_neighbour(self, nbr_vertex, weight=0):
        self.neighbours[nbr_vertex] = weight

    def get_neighbours(self):
        return list(self.neighbours.keys())

    def get_weight(self, neighbour):
        return self.neighbours.get(neighbour, None)

    def __eq__(self, other):
        return self.id_ == other.id_

    def __lt__(self, other):
        return self.id_ < other.id_
    
    def __hash__(self):
        return hash(self.id_)

    def __str__(self):
        neighbours_id = [neighbour.id_ for neighbour in self.get_neighbours()]
        neighbours_str = ", ".join(neighbours_id)
        return ("Vertex {} is connected to: {}".format(self.id_, neighbours_str))

# test cases 
v1 = Vertex("1")
assert v1.id_ == "1" and len(v1.neighbours) == 0
v2 = Vertex("2")
v1.add_neighbour(v2)
assert v1.get_neighbours()[0].id_ == "2" and v1.neighbours[v1.get_neighbours()[0]] == 0
v3 = Vertex("3")
v1.add_neighbour(v3, 3)
assert v1.get_weight(v3) == 3
v4 = Vertex("4")
assert v1.get_weight(v4) == None
assert v1 < v2
assert v1 != v2
assert str(v1) == "Vertex 1 is connected to: 2, 3"

# CS3. Create a class Graph to represent a Graph. The class has the following attribute:
# (1) vertices: which is a dictionary of Vertices. 
#     The keys are the ids of the Vertices and the values are Vertex object instances.

# The class has the following property:
# (1) num_vertices: which is a computed property that returns the number of vertices in the graph.

# The class also has the following methods:
# (1) __init__(self): which initializes the graph with an empty dictionary.
# (2) _create_vertex(self, id_): which creates a new Vertex object with a given id_. 
#     This method is never called directly and is only used by add_vertex(id_).
# (3) add_vertex(self, id_): which creates a new Vertex object, 
#     adding it into the dictionary vertices. The argument id_ is a String. 
#     This method should call _create_vertex(id_).
# (4) get_vertex(self, id_): which returns the Vertex object instance of the requested id_. 
#     The method should return None if the requested id_ cannot be found. The argument id_ is a String.
# (5) add_edge(start_v, end_v): which creates an edge from one Vertex to another Vertex. 
#     The arguments are the id_s of the two vertices and are both Strings.
# (6) get_neighbours(self, id_): which returns a list of id_s all the neighbouring vertices 
#     (of the specified Vertex id_). It should return None if id_ cannot be found. 
#     The argument id_ is a String and the elements of the output list are of str data type.
# (7) __contains__(self, id_): which returns either True or False depending on whether the graph 
#     contains the specified Vertex's id_. The argument id_ is a String.

class Graph:
    def __init__(self):
        self.vertices = {}

    def _create_vertex(self, id_):
        return Vertex(id_)
        
    def add_vertex(self, id_):
        v = self._create_vertex(id_)
        self.vertices[v.id_] = v
    
    def get_vertex(self, id_):
        return self.vertices.get(id_, None)

    def add_edge(self, start_v, end_v, weight=0):
        # start_v and end_v are id_ of vertices, so strings
        # we assign the Vertex object with the specified id_ to a variable. 
        # we use the add_neighbour method from Vertex() class to add the edges with the weight. 
        start_vertex = self.vertices[start_v]
        end_vertex = self.vertices[end_v]
        start_vertex.add_neighbour(end_vertex, weight)
    
    def get_neighbours(self, id_):
        vertex = self.vertices.get(id_)
        return [vert.id_ for vert in vertex.get_neighbours()]

    def __contains__(self, id_):
        return id_ in self.vertices.keys()

    def __iter__(self):
        for k, v in self.vertices.items():
            yield v
    
    @property
    def num_vertices(self):
        return len(self.vertices)

# test cases
g = Graph()
assert g.vertices == {} and g.num_vertices == 0
g.add_vertex("A")
g.add_vertex("B")
g.add_vertex("C")
g.add_vertex("D")
g.add_vertex("E")
g.add_vertex("F")
assert g.num_vertices == 6
assert "A" in g
assert "B" in g
assert "C" in g
assert "D" in g
assert "E" in g
assert "F" in g
g.add_edge("A", "B")
g.add_edge("A", "C")
g.add_edge("B", "C")
g.add_edge("B", "D")
g.add_edge("C", "D")
g.add_edge("D", "C")
g.add_edge("E", "F")
g.add_edge("F", "C")
assert sorted(g.get_neighbours("A")) == ["B", "C"]
assert sorted(g.get_neighbours("B")) == ["C", "D"]
assert sorted(g.get_neighbours("C")) == ["D"]
assert [v.id_ for v in g] == ["A", "B", "C", "D", "E", "F"]

# CS4. Create a subclass of Vertex called VertexSearch. 
# This class has the following additional attributes:
# (1) colour: which is a mark on the vertex during the search algorithm. 
#             It is of String data type and should be set to "white" by default.
# (2) d: which is an Integer denoting the distance from other Vertex to the current Vertex 
#        in Breath-First-Search. This is also used to record discovery time in Depth-First-Search. 
#        This attribute should be initialized to sys.maxsize at the start.
# (3) f: which is an Integer denoting the final time in Depth-First-Search. 
#        This attribute should be initialized to sys.maxsize at the start.
# (4) parent: which is a reference to the parent Vertex object. 
#             This attribute should be set to None at the start.

import sys

class VertexSearch(Vertex):
    
    def __init__(self, id=""):
        super().__init__(id)
        self.colour = "white"
        self.d = sys.maxsize
        self.f = sys.maxsize
        self.parent = None

# test case 
v = VertexSearch()
assert v.id_ == ""
assert v.colour == "white"
assert v.d == sys.maxsize
assert v.f == sys.maxsize
assert v.parent == None
parent_method = getattr(v, 'get_neighbours', None)
assert callable(parent_method)
parent_method = getattr(v, 'get_weight', None)
assert callable(parent_method)

# CS5. You should do this after you completed HW2. 
# Create a class Search2D which takes in an object GraphSearch for its initialization. 
# The class should have the following methods:
# clear_vertices(): which sets the attributes f all the vertices:
# colour to "white"
# d to sys.maxsize
# f to sys.maxsize
# parent to None.

# Class definition: GraphSearch(Graph)
class GraphSearch(Graph):

    def _create_vertex(self, id_):
        return VertexSearch(id_)

# Class definition: Search2D
class Search2D:

    def __init__(self, g):
        self.graph = g

    def clear_vertices(self):
        for vert in self.graph.vertices.values():
            vert.color = "white"
            vert.d = sys.maxsize
            vert.f = sys.maxsize
            vert.parent = None

    def __iter__(self):
        return iter([v for v in self.graph])

    def __len__(self):
        return len([v for v in self.graph.vertices])

# test cases 
g4 = GraphSearch()
g4.add_vertex("A")
g4.add_vertex("B")
g4.add_vertex("C")
g4.add_vertex("D")
g4.add_vertex("E")
g4.add_vertex("F")
g4.add_edge("A", "B")
g4.add_edge("A", "C")
g4.add_edge("B", "C")
g4.add_edge("B", "D")
g4.add_edge("C", "D")
g4.add_edge("D", "C")
g4.add_edge("E", "F")
g4.add_edge("F", "C")
gs4 = Search2D(g4)
gs4.clear_vertices()
assert len(gs4) == 6
assert [v.id_ for v in gs4] == ["A", "B", "C", "D", "E", "F"]
assert [v.colour for v in gs4] == ["white" for v in range(len(gs4))]
assert [v.d for v in gs4] == [sys.maxsize for v in range(len(gs4))]
assert [v.f for v in gs4] == [sys.maxsize for v in range(len(gs4))]
assert [v.parent for v in gs4] == [None for v in range(len(gs4))]

# CS6. Create a class SearchBFS which is a subclass of Search2D. 
# This subclass should implement the Breadth First Search algorithm in the following methods:
# search_from(start): which initializes the d and parent attributes of each vertices in the graph 
#                     from the start Vertex following Breadth-First-Search algorithm. 
#                    Use your previous code that implements Queue data structure.
# get_shortest_path(start, dest): which returns a list of vertex ids that forms a shortest path 
#                                 from Vertex start to Vertex dest. 
#                                 This method should call get_path() (see next method in the list) 
#                                 and pass on an empty list as one of the input arguments. 
#                                 The method get_path() will populate this list if there is a path.
#   If the path list is empty after calling get_path(), 
#   this means that either the starting vertex or the destination vertex do not exist in the grapth. 
#   In this case, exit the function returning a None object.
#   If the path list is not empty, it will either contain No Path as one of the items or 
#   a list of vertices that gives the path from the starting vertex to the destination vertex. 
#   In this case, simply return the list as it is.
# get_path(start, dest, result): which modifies the input list result.
#                                This method should first check whether the starting vertex 
#                                and the destination vertex exist in the grapth. 
#                                If they do not exist in either case, the method should exit 
#                                returning a None object.
#   If the starting and destination vertex exist in the graph, 
#   this method should call search_from() when the distance at start Vertex is not zero. 
#   A non-zero distance at the starting vertex means that we have not run the BFS algorithm 
#   from that starting vertex.
#   if the destination and the starting vertex are the same, modify the result list with this one vertex. 
#   This means that we have found the path that consists of only one vertex.
#   if the destination vertex has no parent, this means there is no path. 
#   Add No Path string into the result list.
#   otherwise, recursively call get_path() and add the result into the result list.

# Class definition: Queue
class Queue:

    def __init__(self):
        self._items = []

    def enqueue(self, value):
        if isinstance(value, int):
            self._items.append(value)

    def dequeue(self):
        if len(self._items) != 0:
            return self._items.pop(0)
        else:
            return None

    def peek(self):
        if len(self._items) != 0:
            return self._items[0]
        else:
            return None

    @property
    def is_empty(self):
        if len(self._items) == 0:
            return True
        else:
            return False

    @property
    def size(self):
        return len(self._items)

# Class definition: SearchBFS(Search2D)
class SearchBFS(Search2D):
    # is the same as super.__init__
    # def __init__(self, g):
    #   self.graph = g

    def search_from(self, start): # start is an id
        # first, reset the vertices attributes (color, d, etc, parent...)
        super.__init__.clear_vertices
        # Updating the attributes of the first vertex.
        startv = self.graph.vertices.get_vertex(start)
        # if start vertex is None, there is no point, you are searching for something not located in the graph. 
        if startv == None:
            return None
        startv.color = "grey"
        startv.d = 0
        startv.parent = None

        # Instantiating the queue
        bfs_queue = Queue()
        # feeding the queue
        bfs_queue.enqueue(start)

        # iterating while the queue is not empty
        while not bfs_queue.is_empty():
            curv = bfs_queue.dequeue()
            for neigh_curv in curv.get_neighbours():
                if neigh_curv.color == "white":
                    neigh_curv.color = "grey"
                    neigh_curv.d = curv.d + 1
                    neigh_curv.parent = curv
                    bfs_queue.enqueue(neigh_curv)
            curv.colour = "black"

    def get_shortest_path(start, dest):
        result = []
        self.get_path(start, dest, result)
        if result = []:
            return None
        else:
            return result
    
    def get_path(start, dest, result):
        vs = self.graph.get_vertex(start)
        ve = self.graph.get_vertex(dest)
        if (vs is None) or (ve is None):
            return None
        if vs.d != 0:
            self.search_from(start)
        if start == dest:
            result.append(start)
        elif ve.parent is None:
            result.append("No Path")
        else:
            self.get_path(start, ve.parent.id_, result)
            result.append(dest)

# test cases
g4 = GraphSearch()
g4.add_vertex("A")
g4.add_vertex("B")
g4.add_vertex("C")
g4.add_vertex("D")
g4.add_vertex("E")
g4.add_vertex("F")
g4.add_edge("A", "B")
g4.add_edge("A", "C")
g4.add_edge("B", "C")
g4.add_edge("B", "D")
g4.add_edge("C", "D")
g4.add_edge("D", "C")
g4.add_edge("E", "F")
g4.add_edge("F", "C")
gs4 = SearchBFS(g4)
gs4.search_from("A")
assert gs4.graph.get_vertex("A").d == 0
assert gs4.graph.get_vertex("A").colour == "black"
assert gs4.graph.get_vertex("A").parent == None
assert gs4.graph.get_vertex("B").d == 1
assert gs4.graph.get_vertex("B").colour == "black"
assert gs4.graph.get_vertex("B").parent == gs4.graph.get_vertex("A")
assert gs4.graph.get_vertex("C").d == 1
assert gs4.graph.get_vertex("C").colour == "black"
assert gs4.graph.get_vertex("C").parent == gs4.graph.get_vertex("A")
assert gs4.graph.get_vertex("D").d == 2
assert gs4.graph.get_vertex("D").colour == "black"
gs4.graph.get_vertex("D").parent
#assert gs4.graph.get_vertex("D").parent == gs4.graph.get_vertex("B")
ans = gs4.get_shortest_path("A", "D")
assert ans == ["A", "B", "D"]
ans = gs4.get_shortest_path("E", "D")
assert ans == ["E", "F", "C", "D"]