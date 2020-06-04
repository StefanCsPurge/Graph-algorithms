import random
import sys
import threading
from GraphDataType import Graph, Activity

fr = open("results.txt", 'w')  # save operation results in file


def readGraphTxt(file,grState):
    f = open(file, 'r')
    line = f.readline()
    parts = line.strip().split()
    n = int(parts[0])  # number of vertices
    m = int(parts[1])  # number of edges
    if m > n*n:
        raise Exception("Too many edges!")
    if grState == 1:  # initial graph
        gr = Graph(n)
    else:  # modified graph
        gr = Graph(n,True)
    lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            x = int(parts[0])
            y = int(parts[1])
            if gr.modified and not gr.isVertex(x):
                gr.addVertex(x)
            if gr.modified and not gr.isVertex(y):
                gr.addVertex(y)
            gr.addEdge(x, y, int(parts[2]))
        elif gr.modified:
            gr.addVertex(parts[0])  # add isolated vertex
    return gr


def readActivitiesGraph(file):
    f = open(file, 'r')
    line = f.readline()
    parts = line.strip().split()
    n = int(parts[0])  # number of activities
    gr = Graph(n+2)
    line = f.readline()
    durations = line.strip().split()
    for i in range(1,n+1):  # now add the prerequisites ( edges )
        line = f.readline()
        parts = line.strip().split()
        if parts[0] != "-1":
            for x in parts:
                gr.addEdge(int(x),i)
        else:
            gr.addEdge(0,i)
    sortedIDs = gr.topologicalSort()
    if sortedIDs is None:
        print("The graph is not a DAG !")
        exit(-1)
    # add activities in topological order
    gr.addActivity(Activity(0, 0, 0, 0, 0, 0))  # add start fictive activity
    for i in range(1,n+1):
        gr.addActivity(Activity(sortedIDs[i],int(durations[sortedIDs[i]-1]),0,0,0,0))
        if not gr.parseNOut(i):
            gr.addEdge(i, n + 1)
    gr.addActivity(Activity(n + 1, 0, 0, 0, 0, 0))  # add end fictive activity
    gr.computeEarliestTimes()
    gr.computeLatestTimes()
    return gr


def writeGraphTxt(g,file):
    f = open(file,'w')
    f.write(f"{g.getVerticesNo()} {g.getEdgesNo()}\n")
    edges = g.getEdges()
    for edge in edges:
        f.write(f"{edge[0]} {edge[1]} {edges[edge]}\n")
    # now print the isolated vertices if the graph was modified
    if g.modified:
        for vertex in g.parseX():
            if not len(g.parseNIn(vertex)) and not len(g.parseNOut(vertex)):
                f.write(f"{vertex}\n")


def createRandomGraph(vert,edg):
    if edg > vert*vert:
        raise Exception("Too many edges!")
    gr = Graph(vert)
    addedEdges = 0
    while addedEdges < edg:
        x = random.randrange(vert)
        y = random.randrange(vert)
        if gr.isEdge(x,y):
            continue
        c = random.randint(-10,22)
        gr.addEdge(x,y,c)
        addedEdges += 1
    return gr


def executeOption(option,g):
    global GCopy
    if option not in range(1,27):
        raise Exception("Non-existent option!")
    if option == 1:
        g.printGraph()
    elif option == 2:
        print(g.getVerticesNo())
        fr.write(f"Vertices number: {g.getVerticesNo()}\n")
    elif option == 3:
        x = int(input("x = "))
        y = int(input("y = "))
        print(g.isEdge(x, y))
        fr.write(f"({x},{y}) edge: {g.isEdge(x,y)}\n")
    elif option == 4:
        x = int(input("Vertex: "))
        print(g.getDegreeIn(x))
        fr.write(f"In degree of vertex {x}: {g.getDegreeIn(x)}\n")
    elif option == 5:
        x = int(input("Vertex: "))
        print(g.getDegreeOut(x))
        fr.write(f"Out degree of vertex {x}: {g.getDegreeOut(x)}\n")
    elif option == 6:
        x = int(input("Vertex: "))
        g.addVertex(x)
        fr.write(f"Vertex {x} added\n")
    elif option == 7:
        x = int(input("Vertex: "))
        g.removeVertex(x)
        fr.write(f"Vertex {x} removed\n")
    elif option == 8:
        x = int(input("x = "))
        y = int(input("y = "))
        c = int(input("c = "))
        g.addEdge(x,y,c)
        fr.write(f"Edge ({x},{y}) with cost={c} added\n")
    elif option == 9:
        x = int(input("x = "))
        y = int(input("y = "))
        g.removeEdge(x,y)
        fr.write(f"Edge ({x},{y}) removed\n")
    elif option == 10:
        file = input("Insert file: ")
        writeGraphTxt(g, file)
    elif option == 11:
        GCopy = g.getGraphCopy()
    elif option == 12:
        if GCopy is None:
            raise Exception("No copy was made!")
        global G
        G = GCopy.getGraphCopy()
    elif option == 13:
        vIterator = g.vertIterator()
        print("The vertices are:")
        fr.write("\nParsing vertices:\n")
        while vIterator.valid():
            print(vIterator.getCurrent())
            fr.write(f"{vIterator.getCurrent()} ")
            vIterator.next()
    elif option == 14:
        x = int(input("Insert vertex: "))
        outEdgesIterator = g.vOutEdgesIterator(x)
        fr.write(f"\nParsing target vertices of {x}:\n")
        while outEdgesIterator.valid():
            print(f"- target vertex {outEdgesIterator.getCurrent()}")
            fr.write(f"- target vertex {outEdgesIterator.getCurrent()}\n")
            outEdgesIterator.next()
    elif option == 15:
        x = int(input("Insert vertex: "))
        inEdgesIterator = g.vInEdgesIterator(x)
        fr.write(f"\nParsing source vertices of {x}:\n")
        while inEdgesIterator.valid():
            print(f"- source vertex {inEdgesIterator.getCurrent()}")
            fr.write(f"- source vertex {inEdgesIterator.getCurrent()}\n")
            inEdgesIterator.next()
    elif option == 16:
        x = int(input("Insert start vertex: "))
        y = int(input("Insert target vertex: "))
        visited, Next = g.BackwardBFS(y,x)
        if x not in visited:
            raise Exception(f"There is no path between {x} and {y}!")
        path = [x]
        while x != y:
            x = Next[x]
            path.append(x)
        print(f"The min length is {len(path)-1}\nThe path is: {path}")
    elif option == 17:
        g.Kosaraju()
    elif option == 18:
        x = int(input("Insert start vertex: "))
        y = int(input("Insert target vertex: "))
        minCost, path = g.lowestCostWalkSlow(x,y)
        print(f"The lowest cost walk is: {minCost}, with path {path}")
    elif option == 19:
        if g.topologicalSort() is None:
            print("The graph is not a DAG !")
        else:
            x = int(input("Insert start vertex: "))
            y = int(input("Insert target vertex: "))
            print(f"The number of lowest cost walks is {g.numberOfLowestCostWalksRecursive(x,y)}")
    elif option == 20:
        x = int(input("Insert start vertex: "))
        y = int(input("Insert target vertex: "))
        c = int(input("Insert newCost: "))
        g.modifyCost(x,y,c)
    elif option == 21:
        sortedActivities = g.getRealActivities()
        if not sortedActivities:
            print("The graph is not a DAG!")
        else:
            print("The (non-fictive) activities in topological order are: ")
            for a in sortedActivities:
                print(a)
    elif option == 22:
        sortedActivities = g.getRealActivities()
        if not sortedActivities:
            print("The graph is not a DAG!")
        else:
            for a in sortedActivities:
                print("Act. " + str(a.ID) + " Earliest start time: " + str(a.eStartT) + " Latest start time: " + str(a.LStartT))
            print("The total time of the project is:",g.getProjectTotalTime())
    elif option == 23:
        sortedActivities = g.getRealActivities()
        if not sortedActivities:
            print("The graph is not a DAG!")
        else:
            print("The critical (non-fictive) activities are: ")
            for a in sortedActivities:
                if a.eStartT == a.LStartT:
                    print(a)
    elif option == 24:
        sortedActivities = g.topologicalSort()
        if sortedActivities is None:
            print("The graph is not a DAG !")
        else:
            print("The graph is a DAG.")
            print(f"The activities in topological order are: {sortedActivities}")
    elif option == 25:
        if g.topologicalSort() is None:
            print("The graph is not a DAG !")
        else:
            x = int(input("Insert start vertex: "))
            y = int(input("Insert target vertex: "))
            print(f"The number of distinct paths is {g.DAG_numberOfDistinctPaths(x, y)}")
    elif option == 26:
        min_V_cover = g.getMinVertexCoverNP()
        print(f"The vertex cover of minimum size is:\n{min_V_cover}\nof size {len(min_V_cover)}")


def run():
    global G
    print("Press 1 to read the graph from the txt file / "
          "2 to generate a random graph / "
          "3 to read activities file")
    choice = int(input(">>"))
    if choice == 1:
        fileName = input("Insert file: ")
        # in the initial graph vertices will be specified as integers from 0 to n-1, where n is the number of vertices
        # in the modified graph there is no such rule and the isolated vertices follow the edge triples in the file
        graphState = int(input("This file contains an initial graph [1] / modified graph [2] ->"))
        G = readGraphTxt(fileName,graphState)
    elif choice == 2:
        v = int(input("No. of vertices: "))
        e = int(input("No. of edges: "))
        G = createRandomGraph(v,e)
    elif choice == 3:
        fileName = input("Insert file: ")
        G = readActivitiesGraph(fileName)

    while True:
        try:
            print("Press:\n"
                  "1  to print the graph in console\n"
                  "2  to get the number of vertices\n"
                  "3  to check if (x,y) is an edge\n"
                  "4  to get the in degree of a specified vertex\n"
                  "5  to get the out degree of a specified vertex\n"
                  "6  to add vertex\n"
                  "7  to remove vertex\n"
                  "8  to add edge\n"
                  "9  to remove edge\n"
                  "10 to write graph to txt file\n"
                  "11 to save a copy of the graph\n"
                  "12 to restore the last saved copy\n"
                  "13 to iterate the set of vertices\n"
                  "14 to iterate the outbound edges of a specified vertex\n"
                  "15 to iterate the inbound edges of a specified vertex\n"
                  "16 to find a lowest length path between 2 vertices\n"
                  "17 to find the strongly-connected components of the graph\n"
                  "18 to find the lowest cost walk between 2 vertices\n"
                  "19 to finds the number of distinct walks of minimum cost between 2 vertices (DAG)\n"
                  "20 to modify cost\n"
                  "21 to display topologically sorted activities\n"
                  "22 to show earliest & latest starting time for each activity & total time of the project\n"
                  "23 to show the critical activities\n"
                  "24 to check DAG and perform topological sorting\n"
                  "25 to find the number of distinct paths between 2 vertices (DAG)\n"
                  "26 to find a vertex cover of minimum size\n"
                  "27 to exit")
            choice = int(input(">>"))
            if choice == 27:
                break
            executeOption(choice,G)
            input("Completed successfully. Press any key to continue...")
        except Exception as ex:
            print(f"Error - {ex}")


sys.setrecursionlimit(4000000)
threading.stack_size(2**27)

if __name__ == "__main__":
    G = None
    GCopy = None
    threading.Thread(target=run).start()
