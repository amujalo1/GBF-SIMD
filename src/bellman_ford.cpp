#include "bellman_ford.h"
#include <iostream>
#include <limits>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include <cstring>
using namespace std;

std::vector<long> runBellmanFordSSSP(Graph* graph, int source_node_id)
{
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    vector<long> node_distances(no_of_nodes, numeric_limits<int>::max() - 100);
    node_distances[source_node_id] = 0;

    // Relax edges
    for (int i = 0; i < no_of_nodes - 1; i++) {
        bool relaxed = false;
        for (int j = 0; j < no_of_edges; j++) {
            int u = graph->edge[j].source;
            int v = graph->edge[j].destination;
            int w = graph->edge[j].weight;

            if (node_distances[u] != numeric_limits<int>::max() &&
                node_distances[u] + w < node_distances[v]) {
                node_distances[v] = node_distances[u] + w;
                relaxed = true;
            }
        }
        if (!relaxed)
            break;
    }

    return node_distances;
}

// cache optimizacija + prefetching
std::vector<long> runBellmanFordSSSP_CACHE(Graph* graph, int source_node_id) {
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    
    vector<long> node_distances(no_of_nodes, numeric_limits<int>::max() - 100);
    node_distances[source_node_id] = 0;
    
    const long MAX_DIST = numeric_limits<int>::max() - 100;
    
    // Sortiranje grana po source čvoru za bolju cache lokalnost
    vector<Edge> sorted_edges(graph->edge, graph->edge + no_of_edges);
    sort(sorted_edges.begin(), sorted_edges.end(), 
         [](const Edge& a, const Edge& b) { return a.source < b.source; });
    
    // Glavna petlja relaksacije
    for (int iter = 0; iter < no_of_nodes - 1; iter++) {
        bool relaxed = false;
        
        // Procesiranje po blokovima za bolju cache iskorištenost
        const int BLOCK_SIZE = 8;
        for (int block_start = 0; block_start < no_of_edges; block_start += BLOCK_SIZE) {
            int block_end = min(block_start + BLOCK_SIZE, no_of_edges);
            
            // Prefetch za sljedeći blok
            if (block_start + BLOCK_SIZE < no_of_edges) {
                __builtin_prefetch(&sorted_edges[block_start + BLOCK_SIZE], 0, 3);
            }
            
            // Procesiranje trenutnog bloka
            for (int j = block_start; j < block_end; j++) {
                int u = sorted_edges[j].source;
                int v = sorted_edges[j].destination;
                int w = sorted_edges[j].weight;
                
                long dist_u = node_distances[u];
                if (dist_u != MAX_DIST) {
                    long new_dist = dist_u + w;
                    if (new_dist < node_distances[v]) {
                        node_distances[v] = new_dist;
                        relaxed = true;
                    }
                }
            }
        }
        
        if (!relaxed) break;
    }
    
    return node_distances;
}

// OpenMP verzija
std::vector<long> runBellmanFordSSSP_OMP(Graph* graph, int source_node_id) {
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    
    vector<long> node_distances(no_of_nodes, numeric_limits<int>::max() - 100);
    node_distances[source_node_id] = 0;
    
    const long MAX_DIST = numeric_limits<int>::max() - 100;
    
    for (int iter = 0; iter < no_of_nodes - 1; iter++) {
        bool relaxed = false;
        
        // Thread-local vektor za detekciju promjena
        #pragma omp parallel
        {
            bool local_relaxed = false;
            
            for (int j = 0; j < no_of_edges; j++) {
                int u = graph->edge[j].source;
                int v = graph->edge[j].destination;
                int w = graph->edge[j].weight;
                
                long dist_u = node_distances[u];
                if (dist_u != MAX_DIST) {
                    long new_dist = dist_u + w;
                    
                    if (new_dist < node_distances[v]) {
                        #pragma omp critical
                        {
                            if (new_dist < node_distances[v]) {
                                node_distances[v] = new_dist;
                                local_relaxed = true;
                            }
                        }
                    }
                }
            }
            
            // Combine local flags
            if (local_relaxed) {
                #pragma omp critical
                relaxed = true;
            }
        }
        
        if (!relaxed) break;
    }
    
    return node_distances;
}

// ============================================================
//      AVX2 Bellman-Ford (SIMD + SoA struktura podataka)
// ============================================================
std::vector<long> runBellmanFordSSSP_AVX2(
    int N, int E, int source_node_id,
    const std::vector<int>& src,
    const std::vector<int>& dst,
    const std::vector<int>& w)
{
    const long INF = std::numeric_limits<int>::max() - 100;

    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    // Sort edges by source
    std::vector<int> order(E);
    for (int i = 0; i < E; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b){
        return src[a] < src[b];
    });

    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;
        int j = 0;

        // SIMD paket od 8 elemenata
        for (; j <= E - 8; j += 8)
        {
            int idx[8];
            for (int k = 0; k < 8; k++) idx[k] = order[j + k];

            // load dist[u] i weights
            long du_arr[8], newdArr[8];
            int vArr[8];
            for (int k = 0; k < 8; k++) {
                du_arr[k] = dist[src[idx[k]]];
                vArr[k] = dst[idx[k]];
            }

            __m256i du = _mm256_loadu_si256((__m256i*)du_arr);
            __m256i wv = _mm256_set_epi32(
                w[idx[7]], w[idx[6]], w[idx[5]], w[idx[4]],
                w[idx[3]], w[idx[2]], w[idx[1]], w[idx[0]]
            );

            __m256i newd = _mm256_add_epi32(du, wv);
            _mm256_storeu_si256((__m256i*)newdArr, newd);

            for (int k = 0; k < 8; k++)
            {
                if (du_arr[k] == INF) continue;

                long v_new = newdArr[k];
                int v = vArr[k];
                if (v_new < dist[v]) {
                    dist[v] = v_new;
                    relaxed = true;
                }
            }
        }

        // Scalar fallback
        for (; j < E; j++)
        {
            int e = order[j];
            int u = src[e];
            int v = dst[e];
            int wt = w[e];
            if (dist[u] != INF) {
                long nd = dist[u] + wt;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    relaxed = true;
                }
            }
        }

        if (!relaxed) break;
    }

    return dist;
}

std::vector<long> runBellmanFordSSSP_AVX512(
    int N, int E, int source_node_id,
    const std::vector<int>& src,
    const std::vector<int>& dst,
    const std::vector<int>& w)
{
    const long INF = std::numeric_limits<long>::max() / 2; // da ne bude overflow prilikom sabiranja

    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    // Sort edges by source
    std::vector<int> order(E);
    for (int i = 0; i < E; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b){
        return src[a] < src[b];
    });

    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;
        int j = 0;

        // SIMD paket od 8 elemenata (AVX-512 može držati 8 x 64-bit int)
        for (; j <= E - 8; j += 8)
        {
            int idx[8];
            for (int k = 0; k < 8; k++) idx[k] = order[j + k];

            long du_arr[8], newdArr[8];
            int vArr[8];
            for (int k = 0; k < 8; k++) {
                du_arr[k] = dist[src[idx[k]]];
                vArr[k] = dst[idx[k]];
            }

            // load dist[u] u AVX-512 register
            __m512i du = _mm512_loadu_si512((__m512i*)du_arr);

            // load weights u 64-bit register
            __m512i wv = _mm512_set_epi64(
                w[idx[7]], w[idx[6]], w[idx[5]], w[idx[4]],
                w[idx[3]], w[idx[2]], w[idx[1]], w[idx[0]]
            );

            // sabiranje: newd = du + wv
            __m512i newd = _mm512_add_epi64(du, wv);
            _mm512_storeu_si512((__m512i*)newdArr, newd);

            // scalar relax (sigurno)
            for (int k = 0; k < 8; k++)
            {
                if (du_arr[k] == INF) continue;

                long v_new = newdArr[k];
                int v = vArr[k];
                if (v_new < dist[v]) {
                    dist[v] = v_new;
                    relaxed = true;
                }
            }
        }

        // Scalar fallback za preostale grane
        for (; j < E; j++)
        {
            int e = order[j];
            int u = src[e];
            int v = dst[e];
            int wt = w[e];
            if (dist[u] != INF) {
                long nd = dist[u] + wt;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    relaxed = true;
                }
            }
        }

        if (!relaxed) break;
    }

    return dist;
}
