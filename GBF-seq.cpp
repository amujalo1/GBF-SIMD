#include <fstream>
#include <map>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <chrono>

using namespace std;

struct Edge {
    int u, v;
    double w;
};

struct MultiEdge {
    int u, v;
    double cost;
    double time;
};

// Standard Bellman-Ford: shortest paths
vector<double> bellmanFordShortestPath(int n, int source, const vector<Edge>& edges) {
    vector<double> dist(n, numeric_limits<double>::infinity());
    dist[source] = 0.0;

    for (int i = 0; i < n - 1; ++i) {
        for (auto& e : edges) {
            if (dist[e.u] + e.w < dist[e.v]) {
                dist[e.v] = dist[e.u] + e.w;
            }
        }
    }
    return dist;
}

// Most reliable path (maximize probability)
vector<double> bellmanFordMostReliablePath(int n, int source, const vector<Edge>& edges) {
    vector<double> reliability(n, 0.0);
    reliability[source] = 1.0;

    for (int i = 0; i < n - 1; ++i) {
        for (auto& e : edges) {
            double candidate = reliability[e.u] * e.w;
            if (candidate > reliability[e.v]) {
                reliability[e.v] = candidate;
            }
        }
    }
    return reliability;
}

// Maximum flow path (maximize minimal capacity along path)
vector<double> bellmanFordMaxFlowPath(int n, int source, const vector<Edge>& edges) {
    vector<double> flow(n, 0.0);
    flow[source] = numeric_limits<double>::infinity();

    for (int i = 0; i < n - 1; ++i) {
        for (auto& e : edges) {
            double candidate = min(flow[e.u], e.w);
            if (candidate > flow[e.v]) {
                flow[e.v] = candidate;
            }
        }
    }
    return flow;
}

// Fuzzy path (max-min composition)
vector<double> bellmanFordFuzzyPath(int n, int source, const vector<Edge>& edges) {
    vector<double> fuzzy(n, 0.0);
    fuzzy[source] = 1.0;

    for (int i = 0; i < n - 1; ++i) {
        for (auto& e : edges) {
            double candidate = min(fuzzy[e.u], e.w);
            if (candidate > fuzzy[e.v]) {
                fuzzy[e.v] = candidate;
            }
        }
    }
    return fuzzy;
}

// Multi-objective Bellman-Ford (cost + time)
vector<pair<double, double>> bellmanFordMultiObjective(int n, int source, const vector<MultiEdge>& edges) {
    vector<pair<double, double>> dist(n, {numeric_limits<double>::infinity(), numeric_limits<double>::infinity()});
    dist[source] = {0.0, 0.0};

    for (int i = 0; i < n - 1; ++i) {
        for (auto& e : edges) {
            double newCost = dist[e.u].first + e.cost;
            double newTime = dist[e.u].second + e.time;
            if (newCost + newTime < dist[e.v].first + dist[e.v].second) {
                dist[e.v] = {newCost, newTime};
            }
        }
    }
    return dist;
}

// Function to load edges from a dataset file
vector<Edge> loadDataset(const string& filename, int& numNodes, vector<int>& indexToNode) {
    vector<Edge> edges;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file " << filename << endl;
        return edges;
    }

    map<int, int> nodeToIndex; // map from original node ID to 0-based index
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int u, v;
        double w;
        if (!(iss >> u >> v >> w)) continue;

        if (nodeToIndex.find(u) == nodeToIndex.end()) {
            nodeToIndex[u] = nodeToIndex.size();
            indexToNode.push_back(u);
        }
        if (nodeToIndex.find(v) == nodeToIndex.end()) {
            nodeToIndex[v] = nodeToIndex.size();
            indexToNode.push_back(v);
        }

        edges.push_back({nodeToIndex[u], nodeToIndex[v], w});
    }
    file.close();
    numNodes = nodeToIndex.size();
    return edges;
}

// Function to automatically find a source node (node with no incoming edges)
int findSourceNode(int numNodes, const vector<Edge>& edges) {
    vector<int> inDegree(numNodes, 0);
    for (auto& e : edges) {
        inDegree[e.v]++;
    }
    // Find first node with inDegree == 0
    for (int i = 0; i < numNodes; ++i) {
        if (inDegree[i] == 0) return i;
    }
    // If all nodes have incoming edges, fallback to 0
    return 0;
}


// Demo main function
int main() {
    int n = 0;
    vector<int> indexToNode;
    vector<Edge> edges = loadDataset("datasets/bfs.txt", n, indexToNode);
    if (edges.empty()) {
        cerr << "Dataset is empty or failed to load." << endl;
        return 1;
    }

    // Map original node ID to 0-based index for source
    int sourceIndex = findSourceNode(n, edges);
    cout << "Automatically selected source node: " << indexToNode[sourceIndex] << "\n";	
    int sourceOriginal = indexToNode[sourceIndex];

    auto start = chrono::high_resolution_clock::now();
    auto shortest = bellmanFordShortestPath(n, sourceIndex, edges);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Shortest paths from node: " << sourceOriginal << "\n";
    for (int i = 0; i < n; ++i)
        cout << "Node " << indexToNode[i] << ": " << shortest[i] << "\n";
    cout << "Execution time: " << duration.count() << " seconds\n\n"; 
 // vector<Edge> reliabilityEdges = { {0,1,0.9}, {0,2,0.7}, {1,3,0.8}, {2,3,0.6} };
 //   start = chrono::high_resolution_clock::now();
 //   auto reliable = bellmanFordMostReliablePath(4,0,reliabilityEdges);
 //   end = chrono::high_resolution_clock::now();
 //   duration = end - start;
 //   cout << "Most reliable paths:\n";
 //   for (int i = 0; i < 4; ++i) cout << "Node " << i << ": " << reliable[i] << "\n";
 //   cout << "Execution time: " << duration.count() << " seconds\n\n";

 //   vector<Edge> flowEdges = { {0,1,5}, {0,2,4}, {1,3,3}, {2,3,2} };
 //   start = chrono::high_resolution_clock::now();
 //   auto maxflow = bellmanFordMaxFlowPath(4,0,flowEdges);
 //   end = chrono::high_resolution_clock::now();
 //   duration = end - start;
 //   cout << "Maximum flow paths:\n";
 //   for (int i = 0; i < 4; ++i) cout << "Node " << i << ": " << maxflow[i] << "\n";
 //   cout << "Execution time: " << duration.count() << " seconds\n\n";

 //   vector<Edge> fuzzyEdges = { {0,1,0.8}, {1,2,0.6}, {0,2,0.5}, {2,3,0.7} };
 //   start = chrono::high_resolution_clock::now();
 //   auto fuzzy = bellmanFordFuzzyPath(4,0,fuzzyEdges);
 //   end = chrono::high_resolution_clock::now();
 //   duration = end - start;
 //   cout << "Fuzzy paths:\n";
 //   for (int i = 0; i < 4; ++i) cout << "Node " << i << ": " << fuzzy[i] << "\n";
 //   cout << "Execution time: " << duration.count() << " seconds\n\n";

   // vector<MultiEdge> multiEdges = { {0,1,3,2}, {1,2,4,5}, {0,2,10,1} };
   // start = chrono::high_resolution_clock::now();
   // auto multi = bellmanFordMultiObjective(3,0,multiEdges);
   // end = chrono::high_resolution_clock::now();
   // duration = end - start;
   // cout << "Multi-objective paths (cost,time):\n";
   // for (int i = 0; i < 3; ++i)
   //     cout << "Node " << i << ": (" << multi[i].first << "," << multi[i].second << ")\n";
   // cout << "Execution time: " << duration.count() << " seconds\n";

    return 0;
}
