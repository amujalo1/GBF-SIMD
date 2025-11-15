#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <string>

struct Edge {
    int source;
    int destination;
    int weight;
};

struct Graph {
    int num_nodes;
    int num_edges;
    Edge* edge;
};

bool createGraph(const std::string& filename,
                 int numNodes,
                 int numEdges,
                 int minW,
                 int maxW);

Graph* readGraph(const std::string& filename);

#endif // GRAPH_UTILS_H
