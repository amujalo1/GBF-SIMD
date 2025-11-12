#include <fstream>
#include <map>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <chrono>
#include <immintrin.h> // AVX intrinsics
#include <omp.h>       // OpenMP

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

// AVX + OpenMP optimizovani Bellman-Ford: shortest paths
vector<double> bellmanFordShortestPath(int n, int source, const vector<Edge>& edges) {
    vector<double> dist(n, numeric_limits<double>::infinity());
    dist[source] = 0.0;

    for (int iter = 0; iter < n - 1; ++iter) {
        // OpenMP paralelizacija - procesiramo SVE ivice
        size_t num_blocks = (edges.size() + 3) / 4; // broj blokova od 4 ivice
        
        #pragma omp parallel for schedule(dynamic, 4)
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t j = b * 4;
            
            if (j + 3 < edges.size()) {
                // AVX procesuira 4 ivice odjednom
                __m256d dist_u = _mm256_set_pd(
                    dist[edges[j+3].u],
                    dist[edges[j+2].u],
                    dist[edges[j+1].u],
                    dist[edges[j].u]
                );
                
                __m256d weights = _mm256_set_pd(
                    edges[j+3].w,
                    edges[j+2].w,
                    edges[j+1].w,
                    edges[j].w
                );
                
                __m256d dist_v = _mm256_set_pd(
                    dist[edges[j+3].v],
                    dist[edges[j+2].v],
                    dist[edges[j+1].v],
                    dist[edges[j].v]
                );
                
                __m256d candidates = _mm256_add_pd(dist_u, weights);
                __m256d mask = _mm256_cmp_pd(candidates, dist_v, _CMP_LT_OQ);
                __m256d result = _mm256_blendv_pd(dist_v, candidates, mask);
                
                double res[4];
                _mm256_storeu_pd(res, result);
                
                // Kritična sekcija za ažuriranje
                #pragma omp critical
                {
                    if (res[0] < dist[edges[j].v]) dist[edges[j].v] = res[0];
                    if (res[1] < dist[edges[j+1].v]) dist[edges[j+1].v] = res[1];
                    if (res[2] < dist[edges[j+2].v]) dist[edges[j+2].v] = res[2];
                    if (res[3] < dist[edges[j+3].v]) dist[edges[j+3].v] = res[3];
                }
            } else {
                // Obradi preostale ivice (manje od 4)
                for (size_t k = j; k < edges.size(); ++k) {
                    auto& e = edges[k];
                    double candidate = dist[e.u] + e.w;
                    #pragma omp critical
                    {
                        if (candidate < dist[e.v]) {
                            dist[e.v] = candidate;
                        }
                    }
                }
            }
        }
    }
    return dist;
}

// AVX + OpenMP optimizovani: Most reliable path
vector<double> bellmanFordMostReliablePath(int n, int source, const vector<Edge>& edges) {
    vector<double> reliability(n, 0.0);
    reliability[source] = 1.0;

    for (int iter = 0; iter < n - 1; ++iter) {
        size_t num_blocks = (edges.size() + 3) / 4;
        
        #pragma omp parallel for schedule(dynamic, 4)
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t j = b * 4;
            
            if (j + 3 < edges.size()) {
                __m256d rel_u = _mm256_set_pd(
                    reliability[edges[j+3].u],
                    reliability[edges[j+2].u],
                    reliability[edges[j+1].u],
                    reliability[edges[j].u]
                );
                
                __m256d probs = _mm256_set_pd(
                    edges[j+3].w,
                    edges[j+2].w,
                    edges[j+1].w,
                    edges[j].w
                );
                
                __m256d rel_v = _mm256_set_pd(
                    reliability[edges[j+3].v],
                    reliability[edges[j+2].v],
                    reliability[edges[j+1].v],
                    reliability[edges[j].v]
                );
                
                __m256d candidates = _mm256_mul_pd(rel_u, probs);
                __m256d mask = _mm256_cmp_pd(candidates, rel_v, _CMP_GT_OQ);
                __m256d result = _mm256_blendv_pd(rel_v, candidates, mask);
                
                double res[4];
                _mm256_storeu_pd(res, result);
                
                #pragma omp critical
                {
                    if (res[0] > reliability[edges[j].v]) reliability[edges[j].v] = res[0];
                    if (res[1] > reliability[edges[j+1].v]) reliability[edges[j+1].v] = res[1];
                    if (res[2] > reliability[edges[j+2].v]) reliability[edges[j+2].v] = res[2];
                    if (res[3] > reliability[edges[j+3].v]) reliability[edges[j+3].v] = res[3];
                }
            } else {
                for (size_t k = j; k < edges.size(); ++k) {
                    auto& e = edges[k];
                    double candidate = reliability[e.u] * e.w;
                    #pragma omp critical
                    {
                        if (candidate > reliability[e.v]) {
                            reliability[e.v] = candidate;
                        }
                    }
                }
            }
        }
    }
    return reliability;
}

// AVX + OpenMP optimizovani: Maximum flow path
vector<double> bellmanFordMaxFlowPath(int n, int source, const vector<Edge>& edges) {
    vector<double> flow(n, 0.0);
    flow[source] = numeric_limits<double>::infinity();

    for (int iter = 0; iter < n - 1; ++iter) {
        size_t num_blocks = (edges.size() + 3) / 4;
        
        #pragma omp parallel for schedule(dynamic, 4)
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t j = b * 4;
            
            if (j + 3 < edges.size()) {
                __m256d flow_u = _mm256_set_pd(
                    flow[edges[j+3].u],
                    flow[edges[j+2].u],
                    flow[edges[j+1].u],
                    flow[edges[j].u]
                );
                
                __m256d capacities = _mm256_set_pd(
                    edges[j+3].w,
                    edges[j+2].w,
                    edges[j+1].w,
                    edges[j].w
                );
                
                __m256d flow_v = _mm256_set_pd(
                    flow[edges[j+3].v],
                    flow[edges[j+2].v],
                    flow[edges[j+1].v],
                    flow[edges[j].v]
                );
                
                __m256d candidates = _mm256_min_pd(flow_u, capacities);
                __m256d result = _mm256_max_pd(candidates, flow_v);
                
                double res[4];
                _mm256_storeu_pd(res, result);
                
                #pragma omp critical
                {
                    if (res[0] > flow[edges[j].v]) flow[edges[j].v] = res[0];
                    if (res[1] > flow[edges[j+1].v]) flow[edges[j+1].v] = res[1];
                    if (res[2] > flow[edges[j+2].v]) flow[edges[j+2].v] = res[2];
                    if (res[3] > flow[edges[j+3].v]) flow[edges[j+3].v] = res[3];
                }
            } else {
                for (size_t k = j; k < edges.size(); ++k) {
                    auto& e = edges[k];
                    double candidate = min(flow[e.u], e.w);
                    #pragma omp critical
                    {
                        if (candidate > flow[e.v]) {
                            flow[e.v] = candidate;
                        }
                    }
                }
            }
        }
    }
    return flow;
}

// AVX + OpenMP optimizovani: Fuzzy path
vector<double> bellmanFordFuzzyPath(int n, int source, const vector<Edge>& edges) {
    vector<double> fuzzy(n, 0.0);
    fuzzy[source] = 1.0;

    for (int iter = 0; iter < n - 1; ++iter) {
        size_t num_blocks = (edges.size() + 3) / 4;
        
        #pragma omp parallel for schedule(dynamic, 4)
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t j = b * 4;
            
            if (j + 3 < edges.size()) {
                __m256d fuzzy_u = _mm256_set_pd(
                    fuzzy[edges[j+3].u],
                    fuzzy[edges[j+2].u],
                    fuzzy[edges[j+1].u],
                    fuzzy[edges[j].u]
                );
                
                __m256d memberships = _mm256_set_pd(
                    edges[j+3].w,
                    edges[j+2].w,
                    edges[j+1].w,
                    edges[j].w
                );
                
                __m256d fuzzy_v = _mm256_set_pd(
                    fuzzy[edges[j+3].v],
                    fuzzy[edges[j+2].v],
                    fuzzy[edges[j+1].v],
                    fuzzy[edges[j].v]
                );
                
                __m256d candidates = _mm256_min_pd(fuzzy_u, memberships);
                __m256d result = _mm256_max_pd(candidates, fuzzy_v);
                
                double res[4];
                _mm256_storeu_pd(res, result);
                
                #pragma omp critical
                {
                    if (res[0] > fuzzy[edges[j].v]) fuzzy[edges[j].v] = res[0];
                    if (res[1] > fuzzy[edges[j+1].v]) fuzzy[edges[j+1].v] = res[1];
                    if (res[2] > fuzzy[edges[j+2].v]) fuzzy[edges[j+2].v] = res[2];
                    if (res[3] > fuzzy[edges[j+3].v]) fuzzy[edges[j+3].v] = res[3];
                }
            } else {
                for (size_t k = j; k < edges.size(); ++k) {
                    auto& e = edges[k];
                    double candidate = min(fuzzy[e.u], e.w);
                    #pragma omp critical
                    {
                        if (candidate > fuzzy[e.v]) {
                            fuzzy[e.v] = candidate;
                        }
                    }
                }
            }
        }
    }
    return fuzzy;
}

// AVX + OpenMP optimizovani: Multi-objective
vector<pair<double, double>> bellmanFordMultiObjective(int n, int source, const vector<MultiEdge>& edges) {
    vector<pair<double, double>> dist(n, {numeric_limits<double>::infinity(), numeric_limits<double>::infinity()});
    dist[source] = {0.0, 0.0};

    for (int iter = 0; iter < n - 1; ++iter) {
        size_t num_blocks = (edges.size() + 3) / 4;
        
        #pragma omp parallel for schedule(dynamic, 4)
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t j = b * 4;
            
            if (j + 3 < edges.size()) {
                __m256d cost_u = _mm256_set_pd(
                    dist[edges[j+3].u].first,
                    dist[edges[j+2].u].first,
                    dist[edges[j+1].u].first,
                    dist[edges[j].u].first
                );
                
                __m256d edge_costs = _mm256_set_pd(
                    edges[j+3].cost,
                    edges[j+2].cost,
                    edges[j+1].cost,
                    edges[j].cost
                );
                
                __m256d new_costs = _mm256_add_pd(cost_u, edge_costs);
                
                __m256d time_u = _mm256_set_pd(
                    dist[edges[j+3].u].second,
                    dist[edges[j+2].u].second,
                    dist[edges[j+1].u].second,
                    dist[edges[j].u].second
                );
                
                __m256d edge_times = _mm256_set_pd(
                    edges[j+3].time,
                    edges[j+2].time,
                    edges[j+1].time,
                    edges[j].time
                );
                
                __m256d new_times = _mm256_add_pd(time_u, edge_times);
                __m256d new_sum = _mm256_add_pd(new_costs, new_times);
                
                __m256d current_sum = _mm256_set_pd(
                    dist[edges[j+3].v].first + dist[edges[j+3].v].second,
                    dist[edges[j+2].v].first + dist[edges[j+2].v].second,
                    dist[edges[j+1].v].first + dist[edges[j+1].v].second,
                    dist[edges[j].v].first + dist[edges[j].v].second
                );
                
                double costs[4], times[4], sums[4], cur_sums[4];
                _mm256_storeu_pd(costs, new_costs);
                _mm256_storeu_pd(times, new_times);
                _mm256_storeu_pd(sums, new_sum);
                _mm256_storeu_pd(cur_sums, current_sum);
                
                #pragma omp critical
                {
                    if (sums[0] < cur_sums[0]) dist[edges[j].v] = {costs[0], times[0]};
                    if (sums[1] < cur_sums[1]) dist[edges[j+1].v] = {costs[1], times[1]};
                    if (sums[2] < cur_sums[2]) dist[edges[j+2].v] = {costs[2], times[2]};
                    if (sums[3] < cur_sums[3]) dist[edges[j+3].v] = {costs[3], times[3]};
                }
            } else {
                for (size_t k = j; k < edges.size(); ++k) {
                    auto& e = edges[k];
                    double newCost = dist[e.u].first + e.cost;
                    double newTime = dist[e.u].second + e.time;
                    #pragma omp critical
                    {
                        if (newCost + newTime < dist[e.v].first + dist[e.v].second) {
                            dist[e.v] = {newCost, newTime};
                        }
                    }
                }
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

    map<int, int> nodeToIndex;
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

// Function to automatically find a source node
int findSourceNode(int numNodes, const vector<Edge>& edges) {
    vector<int> inDegree(numNodes, 0);
    for (auto& e : edges) {
        inDegree[e.v]++;
    }
    for (int i = 0; i < numNodes; ++i) {
        if (inDegree[i] == 0) return i;
    }
    return 0;
}

int main() {
    // Postavi broj OpenMP niti
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " OpenMP threads\n\n";

    int n = 0;
    vector<int> indexToNode;
    vector<Edge> edges = loadDataset("datasets/bfs.txt", n, indexToNode);
    if (edges.empty()) {
        cerr << "Dataset is empty or failed to load." << endl;
        return 1;
    }

    int sourceIndex = findSourceNode(n, edges);
    cout << "Automatically selected source node: " << indexToNode[sourceIndex] << "\n";	
    int sourceOriginal = indexToNode[sourceIndex];

    cout << "=== AVX + OpenMP Optimized Bellman-Ford Algorithms ===\n\n";

    auto start = chrono::high_resolution_clock::now();
    auto shortest = bellmanFordShortestPath(n, sourceIndex, edges);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Shortest paths from node: " << sourceOriginal << "\n";
    for (int i = 0; i < min(10, n); ++i)
        cout << "Node " << indexToNode[i] << ": " << shortest[i] << "\n";
    if (n > 10) cout << "... (showing first 10 nodes)\n";
    cout << "Execution time: " << duration.count() << " seconds\n\n";

    return 0;
}
