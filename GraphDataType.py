from copy import deepcopy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cmath import inf


def printMatrix(m, D):
    print(f"D{m+1}")
    for row in D:
        for val in row:
            print('{:4}'.format(val),end=' ')
        print()
    print()


class Activity:
    def __init__(self,ID,duration,earliestStartTime,earliestEndTime,latestStartTime,latestEndTime):
        self.ID = ID
        self.duration = duration
        self.eStartT = earliestStartTime
        self.eEndT = earliestEndTime
        self.LStartT = latestStartTime
        self.LEndT = latestEndTime

    def __str__(self):
        return "Act. " + str(self.ID) + " - Duration " + str(self.duration) + \
                " - Earliest start time " + str(self.eStartT) + " - Earliest end time " + str(self.eEndT) + \
                " - Latest start time " + str(self.LStartT) + " - Latest end time " + str(self.LEndT)


class Graph:
    """A directed graph, represented as three dictionaries,
    one from each vertex to the set of outbound neighbours,
    the other from each vertex to the set of inbound neighbours
    and the last from each edge to its costs"""

    def __init__(self, n, modified=False):
        """Creates a graph with n vertices (numbered from 0 to n-1)
        and no edges, by default (if not modified) """
        self._dictOut = {}
        self._dictIn = {}
        self._edgeCosts = {}
        self.modified = modified
        self.counter = 0
        self.min = 0
        self.minSubset = []
        if not modified:
            for i in range(n):
                self._dictOut[i] = []
                self._dictIn[i] = []
        self._activities = []

    def getVerticesNo(self):
        """Gets the number of vertices."""
        return len(self._dictIn)

    def getEdgesNo(self):
        """Gets the number of edges."""
        return len(self._edgeCosts)

    def getEdges(self):
        """Gets all the edges (with their costs) (deepcopy)."""
        return deepcopy(self._edgeCosts)

    def parseX(self):
        """Returns an iterable containing all the vertices"""
        return list(self._dictOut.keys())

    def parseNOut(self, x):
        """Returns an iterable containing the outbound neighbours of x"""
        return self._dictOut[x]

    def parseNIn(self, x):
        """Returns an iterable containing the inbound neighbours of x"""
        return self._dictIn[x]

    def isEdge(self, x, y):
        """Returns True if there is an edge from x to y, False otherwise"""
        return y in self._dictOut[x]

    def isVertex(self, x):
        """Checks if x is a vertex"""
        return x in self._dictOut

    def addEdge(self, x, y, c=0):
        """Adds an edge from x to y.
        Precondition: there is no edge from x to y"""
        if self.isEdge(x,y):
            raise Exception("Already existing edge!")
        self._dictOut[x].append(y)
        self._dictIn[y].append(x)
        self._edgeCosts[(x, y)] = c
        self.modified = True

    def modifyCost(self,x,y,newC):
        if not self.isEdge(x,y):
            raise Exception("Non existing edge!")
        self._edgeCosts[(x, y)] = newC

    def removeEdge(self, x, y):
        """Removes the edge from x to y
        Precondition: there is an edge from x to y"""
        if not self.isEdge(x,y):
            raise Exception("Non-existent edge!")
        self._dictOut[x].remove(y)
        self._dictIn[y].remove(x)
        del self._edgeCosts[(x,y)]
        self.modified = True

    def addVertex(self, x):
        """Adds the vertex x"""
        if self.isVertex(x):
            raise Exception("Vertex already exists!")
        self._dictOut[x] = []
        self._dictIn[x] = []
        self.modified = True

    def removeVertex(self, x):
        """Removes the vertex x
        Precondition: there is a vertex x"""
        for y in self._dictOut[x]:
            del self._edgeCosts[(x,y)]
            self._dictIn[y].remove(x)
        del self._dictOut[x]
        for y in self._dictIn[x]:
            if x != y:
                del self._edgeCosts[(y,x)]
                self._dictOut[y].remove(x)
        del self._dictIn[x]
        self.modified = True

    def addActivity(self,a):
        self._activities.append(a)

    def getActivity(self, ID):
        for a in self._activities:
            if a.ID == ID:
                return a
        return None

    def getRealActivities(self):
        return self._activities[1:-1]

    def getProjectTotalTime(self):
        return self._activities[-1].eEndT

    def getEdgeCost(self, x, y):
        """Gets the cost of the edge from x to y
        Precondition: there is an edge from x to y"""
        return self._edgeCosts[(x, y)]

    def setEdgeCost(self, x, y, newCost):
        """Sets the cost of the edge from x to y"""
        self._edgeCosts[(x, y)] = newCost

    def getDegreeIn(self, vertex):
        """Gets the in degree of a vertex"""
        return len(self._dictIn[vertex])

    def getDegreeOut(self, vertex):
        """Gets the out degree of a vertex"""
        return len(self._dictOut[vertex])

    def getGraphCopy(self):
        """Gets a deepcopy of the graph"""
        return deepcopy(self)

    def printGraph(self):
        """Prints the graph, each edge on a line in the format:
        x --(edgeCost)--> y"""
        edgesDict = {}
        fromList = []
        toList = []
        for x in self._dictIn.keys():
            if len(self._dictOut[x]):
                for y in self._dictOut[x]:
                    print(f"{x} --({self._edgeCosts[(x,y)]})--> {y}")
                    if self.getVerticesNo() < 100 and self.getEdgesNo() < 55:
                        fromList.append(x)
                        toList.append(y)
            else:
                print(x)
                if self.getVerticesNo() < 100 and self.getEdgesNo() < 55:
                    fromList.append(x)
                    toList.append(x)
            if len(fromList):
                edgesDict['from'] = fromList
                edgesDict['to'] = toList
                df = pd.DataFrame(edgesDict)
                G = nx.from_pandas_edgelist(df,'from','to',create_using=nx.DiGraph())
                nx.draw(G,with_labels=True,node_size=250,alpha=0.8,arrows=True)
                plt.show()

    def vertIterator(self):
        """Returns an iterator for the vertices"""
        return Iterator(self.parseX())

    def vOutEdgesIterator(self,x):
        """Returns an iterator for the out edges of the vertex x"""
        return Iterator(self.parseNOut(x))

    def vInEdgesIterator(self,x):
        """Returns an iterator for the in edges of the vertex x"""
        return Iterator(self.parseNIn(x))

    def BackwardBFS(self,t,s):
        """
        Input:
        :param s: start vertex
        :param t: target vertex
        Output:
        accessible (visited) : the set of vertices that are accessible from t
        Next : a map that maps each accessible vertex to its predecessor on a path from t to it
        :return: tuple of accessible and prev
        """
        queue = [t]
        Next = {}
        # dist = {t:0}
        visited = {t}
        while len(queue):
            y = queue.pop(0)
            for x in self._dictIn[y]:
                if x not in visited:
                    queue.append(x)
                    visited.add(x)
                    # dist[x] = dist[y] + 1
                    Next[x] = y
                if x == s:
                    return visited, Next
        # return visited, Next

    def DF1(self,vertex,visited,processed):
        for y in self._dictOut[vertex]:
            if y not in visited:
                visited.add(y)
                self.DF1(y,visited,processed)
        processed.append(vertex)

    def Kosaraju(self):
        """Finds the strongly-connected components of the directed graph in O(n+m)"""
        processed = []  # stack of processed vertices
        visited = set()
        for s in self.parseX():
            if s not in visited:
                visited.add(s)
                self.DF1(s,visited,processed)
        visited.clear()
        queue = []
        c = 0  # counter for the strongly connected components
        while len(processed):
            s = processed.pop()
            if s not in visited:
                c += 1
                component = [s]
                queue.append(s)
                visited.add(s)
                while len(queue):
                    x = queue.pop(0)
                    for y in self._dictIn[x]:  # similar to backward BFS
                        if y not in visited:
                            visited.add(y)
                            queue.append(y)
                            component.append(y)
                print(f"Strongly-connected component {c}: ",component)

    def getAdjacencyMatrix(self):
        n = self.getVerticesNo()
        W = [[0.0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.isEdge(i,j):
                        W[i][j] = self.getEdgeCost(i,j)
                    else:
                        W[i][j] = inf
        return W

    def getNextMatrix(self):
        n = self.getVerticesNo()
        Next = [[-1 for i in range(n)] for j in range(n)]
        for edge in self.getEdges():
            Next[edge[0]][edge[1]] = edge[1]
        for i in range(n):
            Next[i][i] = i
        return Next

    @staticmethod
    def getPath(Next,s,t):
        if Next[s][t] == -1:
            return []
        path = [s]
        while s != t:
            s = Next[s][t]
            path.append(s)
        return path

    def lowestCostWalkSlow(self,x,y):
        W = self.getAdjacencyMatrix()
        D = deepcopy(W)
        Next = self.getNextMatrix()
        printMatrix(0,D)
        n = self.getVerticesNo()
        for m in range(1,n-1):
            D = self.extend(D,W,Next)
            printMatrix(m,D)
        return D[x][y],self.getPath(Next,x,y)

    def extend(self,D,W,Next):
        n = self.getVerticesNo()
        D_copy = deepcopy(D)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    p = D_copy[i][k] + W[k][j]
                    if D[i][j] > p:
                        D[i][j] = p
                        Next[i][j] = Next[i][k]
                if i == j and D[i][i] < 0:
                    raise Exception("Negative cost cycle!")
        return D

    def Dijkstra(self,s,t):
        """
        Input:
        self : directed graph with costs
        s, t : two vertices
        Output:
        dist : a map that associates, to each accessible vertex, the cost of the minimum
            cost walk from s to it
        prev : a map that maps each accessible vertex to its predecessor on a path from s to it
        """
        q = PriorityQueue()
        prev = {}
        dist = {}
        q.add(s,0)
        dist[s] = 0
        found = False
        while not q.isEmpty() and not found:
            x = q.pop()
            for y in self._dictOut[x]:
                if y not in dist.keys() or dist[x] + self.getEdgeCost(x,y) < dist[y]:
                    dist[y] = dist[x] + self.getEdgeCost(x,y)
                    q.add(y,dist[y])
                    prev[y] = x
            if x == t:
                found = True
        return dist[t]

    def numberOfLowestCostWalksRecursive(self,x,y):
        minCost = self.Dijkstra(x,y)
        print(f"The minimum walk cost is {minCost}")
        c_sum = 0
        visited = {x}
        self.counter = 0
        self.countLowestCostWalks(x,y,minCost,c_sum,visited)  # using DFS algorithm
        return self.counter

    def countLowestCostWalks(self,x,end,minCost,c_sum,visited):
        if x == end:
            if c_sum == minCost:
                self.counter += 1
                print(visited)
            return
        for y in self._dictOut[x]:
            if y not in visited:
                visited.add(y)
                c = self.getEdgeCost(x,y)
                c_sum += c
                self.countLowestCostWalks(y,end,minCost,c_sum,visited)
                visited.remove(y)
                c_sum -= c

    def DAG_numberOfDistinctPaths(self,x,y):
        visited = {x}
        self.counter = 0
        self.countDistinctWalksDFS(x, y, visited)
        return self.counter

    def countDistinctWalksDFS(self,x,end,visited):
        if x == end:
            self.counter += 1
            print(visited)
            return
        for y in self._dictOut[x]:
            if y not in visited:
                visited.add(y)
                self.countDistinctWalksDFS(y,end,visited)
                visited.remove(y)

    def TopoSortDFS(self,x,Sorted,fullyProcessed,inProcess):
        inProcess.add(x)
        for y in self.parseNIn(x):
            if y in inProcess:
                return False
            else:
                if y not in fullyProcessed:
                    ok = self.TopoSortDFS(y, Sorted, fullyProcessed, inProcess)
                    if not ok:
                        return False
        inProcess.remove(x)
        Sorted.append(x)
        fullyProcessed.add(x)
        return True

    def topologicalSort(self):   # sort the vertices in topological order
        Sorted = []
        fullyProcessed = set()
        inProcess = set()
        for x in self.parseX():
            if x not in fullyProcessed:
                ok = self.TopoSortDFS(x, Sorted, fullyProcessed, inProcess)
                if not ok:
                    return None
        return Sorted

    def computeEarliestTimes(self):
        for i in range(1,len(self._activities)):
            act = self._activities[i]
            if self.parseNIn(act.ID):
                act.eStartT = max([self.getActivity(x).eEndT for x in self.parseNIn(act.ID)])
            else:
                act.eStartT = 0
            act.eEndT = act.eStartT + act.duration

    def computeLatestTimes(self):
        n = len(self._activities) - 2
        self._activities[n+1].LEndT = self._activities[n + 1].LStartT = self._activities[n + 1].eEndT
        for i in range(n,-1,-1):
            act = self._activities[i]
            if self.parseNOut(act.ID):
                act.LEndT = min([self.getActivity(y).LStartT for y in self.parseNOut(act.ID)])
            else:
                act.LEndT = self._activities[n + 1].LStartT
            act.LStartT = act.LEndT - act.duration

    def getVertexCover(self):
        visited = {}
        vertices = []
        for x in self.parseX():
            visited[x] = False

        for x in self.parseX():
            if not visited[x]:
                for y in self.parseX():
                    if y in self.parseNOut(x) or y in self.parseNIn(x):
                        if not visited[y]:
                            visited[y] = True
                            visited[x] = True
                            break
        for x in self.parseX():
            if visited[x]:
                vertices.append(x)
        return vertices

    def getMinVertexCoverNP(self):
        vertices = deepcopy(self.parseX())  # take the set of all vertices
        self.min = self.getVerticesNo() + 1
        subset = []
        self.minSubset = []
        self._subsetsUtil(vertices,subset,0)
        return self.minSubset

    def _subsetsUtil(self, A, subset, index):
        coveredEdges = []     # check subset
        for v in subset:
            for x in self.parseX():
                if self.isEdge(v,x) or self.isEdge(x,v):
                    coveredEdges.append((v,x))  # undirected graph
                    coveredEdges.append((x,v))
        ok = True
        allEdges = list(self.getEdges().keys())
        for edge in allEdges:
            if edge not in coveredEdges:
                ok = False
                break
        if ok and len(subset) and len(subset) < self.min:
            self.min = len(subset)
            self.minSubset = deepcopy(subset)

        for i in range(index, len(A)):
            # include the A[i] in subset.
            subset.append(A[i])
            # move onto the next element.
            self._subsetsUtil(A, subset, i + 1)
            # exclude the A[i] from subset and
            # triggers backtracking.
            subset.pop(-1)
        return


class Iterator:
    def __init__(self,iterable):
        """Creates an iterator for a given iterable"""
        self.__it = iterable
        self.__idx = 0

    def valid(self):
        """Checks if the current index has reached the end"""
        if self.__idx == len(self.__it):
            return False
        return True

    def first(self):
        """Sets the current index for the first element"""
        self.__idx = 0

    def next(self):
        """Gets the index for the next element"""
        if not self.valid():
            raise Exception("You have reached the end!")
        self.__idx += 1

    def getCurrent(self):
        """Gets the current element"""
        if self.__idx < len(self.__it):
            return self.__it[self.__idx]
        raise ValueError()


class PriorityQueue:
    def __init__(self):
        self.__values = {}

    def isEmpty(self):
        return len(self.__values) == 0

    def pop(self):
        topPriority = None
        topObject = None
        for obj in self.__values:
            objPriority = self.__values[obj]
            if topPriority is None or topPriority > objPriority:
                topPriority = objPriority
                topObject = obj
        del self.__values[topObject]
        return topObject

    def add(self, obj, priority):
        self.__values[obj] = priority

    def contains(self, val):
        return val in self.__values
