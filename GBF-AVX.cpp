#include <fstream>
#include <map>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <chrono>
#include <immintrin.h> // AVX intrinsics

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

// AVX-optimizovani Bellman-Ford: shortest paths
vector<double> bellmanFordShortestPath(int n, int source, const vector<Edge>& edges) {
    vector<double> dist(n, numeric_limits<double>::infinity());
    dist[source] = 0.0;

    for (int i = 0; i < n - 1; ++i) {
        // Procesiramo ivice u blokovima od 4 (AVX procesuira 4 double vrijednosti)
        size_t j = 0;
        for (; j + 3 < edges.size(); j += 4) {
            // Učitaj dist[u] za 4 ivice
            __m256d dist_u = _mm256_set_pd(
                dist[edges[j+3].u],
                dist[edges[j+2].u],
                dist[edges[j+1].u],
                dist[edges[j].u]
            );
            
            // Učitaj težine ivica
            __m256d weights = _mm256_set_pd(
                edges[j+3].w,
                edges[j+2].w,
                edges[j+1].w,
                edges[j].w
            );
            
            // Učitaj trenutne dist[v] vrijednosti
            __m256d dist_v = _mm256_set_pd(
                dist[edges[j+3].v],
                dist[edges[j+2].v],
                dist[edges[j+1].v],
                dist[edges[j].v]
            );
            
            // Izračunaj candidate = dist[u] + w
            __m256d candidates = _mm256_add_pd(dist_u, weights);
            
            // Uporedi i ažuriraj (minimum)
            __m256d mask = _mm256_cmp_pd(candidates, dist_v, _CMP_LT_OQ);
            __m256d result = _mm256_blendv_pd(dist_v, candidates, mask);
            
            // Izvuci rezultate i ažuriraj
            double res[4];
            _mm256_storeu_pd(res, result);
            
            if (res[0] < dist[edges[j].v]) dist[edges[j].v] = res[0];
            if (res[1] < dist[edges[j+1].v]) dist[edges[j+1].v] = res[1];
            if (res[2] < dist[edges[j+2].v]) dist[edges[j+2].v] = res[2];
            if (res[3] < dist[edges[j+3].v]) dist[edges[j+3].v] = res[3];
        }
        
        // Obradi preostale ivice
        for (; j < edges.size(); ++j) {
            auto& e = edges[j];
            if (dist[e.u] + e.w < dist[e.v]) {
                dist[e.v] = dist[e.u] + e.w;
            }
        }
    }
    return dist;
}

// AVX-optimizovani: Most reliable path (maximize probability)
vector<double> bellmanFordMostReliablePath(int n, int source, const vector<Edge>& edges) {
    vector<double> reliability(n, 0.0);
    reliability[source] = 1.0;

    for (int i = 0; i < n - 1; ++i) {
        size_t j = 0;
        for (; j + 3 < edges.size(); j += 4) {
            // Učitaj reliability[u]
            __m256d rel_u = _mm256_set_pd(
                reliability[edges[j+3].u],
                reliability[edges[j+2].u],
                reliability[edges[j+1].u],
                reliability[edges[j].u]
            );
            
            // Učitaj vjerovatnoće (težine)
            __m256d probs = _mm256_set_pd(
                edges[j+3].w,
                edges[j+2].w,
                edges[j+1].w,
                edges[j].w
            );
            
            // Učitaj trenutne reliability[v]
            __m256d rel_v = _mm256_set_pd(
                reliability[edges[j+3].v],
                reliability[edges[j+2].v],
                reliability[edges[j+1].v],
                reliability[edges[j].v]
            );
            
            // candidate = reliability[u] * prob
            __m256d candidates = _mm256_mul_pd(rel_u, probs);
            
            // Uporedi i ažuriraj (maksimum)
            __m256d mask = _mm256_cmp_pd(candidates, rel_v, _CMP_GT_OQ);
            __m256d result = _mm256_blendv_pd(rel_v, candidates, mask);
            
            double res[4];
            _mm256_storeu_pd(res, result);
            
            if (res[0] > reliability[edges[j].v]) reliability[edges[j].v] = res[0];
            if (res[1] > reliability[edges[j+1].v]) reliability[edges[j+1].v] = res[1];
            if (res[2] > reliability[edges[j+2].v]) reliability[edges[j+2].v] = res[2];
            if (res[3] > reliability[edges[j+3].v]) reliability[edges[j+3].v] = res[3];
        }
        
        for (; j < edges.size(); ++j) {
            auto& e = edges[j];
            double candidate = reliability[e.u] * e.w;
            if (candidate > reliability[e.v]) {
                reliability[e.v] = candidate;
            }
        }
    }
    return reliability;
}

// AVX-optimizovani: Maximum flow path
vector<double> bellmanFordMaxFlowPath(int n, int source, const vector<Edge>& edges) {
    vector<double> flow(n, 0.0);
    flow[source] = numeric_limits<double>::infinity();

    for (int i = 0; i < n - 1; ++i) {
        size_t j = 0;
        for (; j + 3 < edges.size(); j += 4) {
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
            
            // candidate = min(flow[u], capacity)
            __m256d candidates = _mm256_min_pd(flow_u, capacities);
            
            // Uporedi i ažuriraj (maksimum)
            __m256d result = _mm256_max_pd(candidates, flow_v);
            
            double res[4];
            _mm256_storeu_pd(res, result);
            
            if (res[0] > flow[edges[j].v]) flow[edges[j].v] = res[0];
            if (res[1] > flow[edges[j+1].v]) flow[edges[j+1].v] = res[1];
            if (res[2] > flow[edges[j+2].v]) flow[edges[j+2].v] = res[2];
            if (res[3] > flow[edges[j+3].v]) flow[edges[j+3].v] = res[3];
        }
        
        for (; j < edges.size(); ++j) {
            auto& e = edges[j];
            double candidate = min(flow[e.u], e.w);
            if (candidate > flow[e.v]) {
                flow[e.v] = candidate;
            }
        }
    }
    return flow;
}

// AVX-optimizovani: Fuzzy path (max-min composition)
vector<double> bellmanFordFuzzyPath(int n, int source, const vector<Edge>& edges) {
    vector<double> fuzzy(n, 0.0);
    fuzzy[source] = 1.0;

    for (int i = 0; i < n - 1; ++i) {
        size_t j = 0;
        for (; j + 3 < edges.size(); j += 4) {
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
            
            // candidate = min(fuzzy[u], membership)
            __m256d candidates = _mm256_min_pd(fuzzy_u, memberships);
            
            // Ažuriraj (maksimum)
            __m256d result = _mm256_max_pd(candidates, fuzzy_v);
            
            double res[4];
            _mm256_storeu_pd(res, result);
            
            if (res[0] > fuzzy[edges[j].v]) fuzzy[edges[j].v] = res[0];
            if (res[1] > fuzzy[edges[j+1].v]) fuzzy[edges[j+1].v] = res[1];
            if (res[2] > fuzzy[edges[j+2].v]) fuzzy[edges[j+2].v] = res[2];
            if (res[3] > fuzzy[edges[j+3].v]) fuzzy[edges[j+3].v] = res[3];
        }
        
        for (; j < edges.size(); ++j) {
            auto& e = edges[j];
            double candidate = min(fuzzy[e.u], e.w);
            if (candidate > fuzzy[e.v]) {
                fuzzy[e.v] = candidate;
            }
        }
    }
    return fuzzy;
}

// AVX-optimizovani: Multi-objective (cost + time)
vector<pair<double, double>> bellmanFordMultiObjective(int n, int source, const vector<MultiEdge>& edges) {
    vector<pair<double, double>> dist(n, {numeric_limits<double>::infinity(), numeric_limits<double>::infinity()});
    dist[source] = {0.0, 0.0};

    for (int i = 0; i < n - 1; ++i) {
        size_t j = 0;
        for (; j + 3 < edges.size(); j += 4) {
            // Cost komponenta
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
            
            // Time komponenta
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
            
            // Suma novih vrijednosti
            __m256d new_sum = _mm256_add_pd(new_costs, new_times);
            
            // Suma trenutnih vrijednosti
            __m256d current_sum = _mm256_set_pd(
                dist[edges[j+3].v].first + dist[edges[j+3].v].second,
                dist[edges[j+2].v].first + dist[edges[j+2].v].second,
                dist[edges[j+1].v].first + dist[edges[j+1].v].second,
                dist[edges[j].v].first + dist[edges[j].v].second
            );
            
            // Uporedi
            __m256d mask = _mm256_cmp_pd(new_sum, current_sum, _CMP_LT_OQ);
            
            double costs[4], times[4], sums[4], cur_sums[4];
            _mm256_storeu_pd(costs, new_costs);
            _mm256_storeu_pd(times, new_times);
            _mm256_storeu_pd(sums, new_sum);
            _mm256_storeu_pd(cur_sums, current_sum);
            
            if (sums[0] < cur_sums[0]) dist[edges[j].v] = {costs[0], times[0]};
            if (sums[1] < cur_sums[1]) dist[edges[j+1].v] = {costs[1], times[1]};
            if (sums[2] < cur_sums[2]) dist[edges[j+2].v] = {costs[2], times[2]};
            if (sums[3] < cur_sums[3]) dist[edges[j+3].v] = {costs[3], times[3]};
        }
        
        for (; j < edges.size(); ++j) {
            auto& e = edges[j];
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

    cout << "=== AVX-Optimized Bellman-Ford Algorithms ===\n\n";

    auto start = chrono::high_resolution_clock::now();
    auto shortest = bellmanFordShortestPath(n, sourceIndex, edges);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Shortest paths from node: " << sourceOriginal << "\n";
    for (int i = 0; i < n; ++i)
        cout << "Node " << indexToNode[i] << ": " << shortest[i] << "\n";
    cout << "Execution time: " << duration.count() << " seconds\n\n";

    return 0;
}
