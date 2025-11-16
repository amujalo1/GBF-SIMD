#include "bellman_ford.h"
#include <iostream>
#include <limits>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include <cstring>
#include <atomic>
using namespace std;

// ============================================================
//      ORIGINALNA VERZIJA (ISPRAVLJENA)
// ============================================================
std::vector<long> runBellmanFordSSSP(Graph* graph, int source_node_id)
{
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    
    // ISPRAVKA: Koristiti long::max umesto int::max
    const long INF = numeric_limits<long>::max() / 2;
    vector<long> node_distances(no_of_nodes, INF);
    node_distances[source_node_id] = 0;

    // Relax edges
    for (int i = 0; i < no_of_nodes - 1; i++) {
        bool relaxed = false;
        for (int j = 0; j < no_of_edges; j++) {
            int u = graph->edge[j].source;
            int v = graph->edge[j].destination;
            long w = graph->edge[j].weight; // long umesto int
            
            // ISPRAVKA: Pravilna provera za INF
            if (node_distances[u] < INF &&
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

// ============================================================
//      CACHE OPTIMIZOVANA VERZIJA (ISPRAVLJENA)
// ============================================================
std::vector<long> runBellmanFordSSSP_CACHE(Graph* graph, int source_node_id) {
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    
    const long INF = numeric_limits<long>::max() / 2;
    vector<long> node_distances(no_of_nodes, INF);
    node_distances[source_node_id] = 0;
    
    // Sortiranje grana po source čvoru za bolju cache lokalnost
    vector<Edge> sorted_edges(graph->edge, graph->edge + no_of_edges);
    sort(sorted_edges.begin(), sorted_edges.end(), 
         [](const Edge& a, const Edge& b) { return a.source < b.source; });
    
    // Glavna petlja relaksacije
    for (int iter = 0; iter < no_of_nodes - 1; iter++) {
        bool relaxed = false;
        
        // Procesiranje po blokovima za bolju cache iskorištenost
        const int BLOCK_SIZE = 64;
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
                long w = sorted_edges[j].weight;
                
                long dist_u = node_distances[u];
                if (dist_u < INF) {
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

// ============================================================
//      OpenMP VERZIJA SA ATOMIC (THREAD-SAFE + OPTIMIZOVANA)
// ============================================================
std::vector<long> runBellmanFordSSSP_OMP(Graph* graph, int source_node_id)
{
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;

    // -----------------------------
    // 1) Sortiranje grana po source
    // -----------------------------
    std::vector<Edge> edges(graph->edge, graph->edge + no_of_edges);

    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) {
                  return a.source < b.source;
              });

    // -----------------------------
    // 2) Distance niz
    // -----------------------------
    std::vector<long> dist(no_of_nodes, LONG_MAX - 1000);
    dist[source_node_id] = 0;

    bool updated = true;

    // -----------------------------
    // 3) RELAKSACIJE (N-1 puta)
    // -----------------------------
    for (int iter = 0; iter < no_of_nodes - 1 && updated; iter++)
    {
        updated = false;

        #pragma omp parallel
        {
            bool local_update = false;

            #pragma omp for schedule(static)
            for (int e = 0; e < no_of_edges; e++)
            {
                // Prefetch unaprijed 16 edge struktura (~1-2 cache line)
                if (e + 16 < no_of_edges)
                    __builtin_prefetch(&edges[e + 16], 0, 1);

                int u = edges[e].source;
                int v = edges[e].destination;
                long w = edges[e].weight;

                long du = dist[u];
                long nd = du + w;

                long oldv = dist[v];

                if (nd < oldv)
                {
                    #pragma omp atomic write
                    dist[v] = nd;
                    local_update = true;
                }
            }

            // Ako je thread našao update → globalni flag
            if (local_update)
            {
                #pragma omp atomic write
                updated = true;
            }
        } // kraj parallel region
    }

    return dist;
}


// ============================================================
//      AVX2 VERZIJA SA CACHE OPTIMIZACIJOM
// ============================================================
std::vector<long> runBellmanFordSSSP_AVX2(
    int N, int E, int source_node_id,
    const std::vector<int>& src,
    const std::vector<int>& dst,
    const std::vector<int>& w)
{
    const long INF = std::numeric_limits<long>::max() / 2;

    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    // Kreiraj strukturu grana za sortiranje
    struct EdgeTmp { int u, v, weight; };
    std::vector<EdgeTmp> edges(E);
    for (int i = 0; i < E; i++) {
        edges[i] = { src[i], dst[i], w[i] };
    }

    // Sortiranje po source čvoru radi cache lokalnosti
    std::sort(edges.begin(), edges.end(), [](const EdgeTmp &a, const EdgeTmp &b){
        return a.u < b.u;
    });

    const int BLOCK_SIZE = 64; // blokiranje radi cache-a

    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;
        for (int block_start = 0; block_start < E; block_start += BLOCK_SIZE)
        {
            int block_end = std::min(block_start + BLOCK_SIZE, E);

            if (block_start + BLOCK_SIZE < E) {
                __builtin_prefetch(&edges[block_start + BLOCK_SIZE], 0, 3);
            }

            int j = block_start;

            // SIMD AVX2 processing - 4 grane po iteraciji
            for (; j + 3 < block_end; j += 4)
            {
                alignas(32) long du[4] = { dist[edges[j+0].u], dist[edges[j+1].u],
                                            dist[edges[j+2].u], dist[edges[j+3].u] };
                alignas(32) long wt[4] = { (long)edges[j+0].weight, (long)edges[j+1].weight,
                                            (long)edges[j+2].weight, (long)edges[j+3].weight };

                __m256i v_du = _mm256_load_si256((__m256i*)du);
                __m256i v_wt = _mm256_load_si256((__m256i*)wt);
                __m256i v_new = _mm256_add_epi64(v_du, v_wt);

                alignas(32) long new_dist[4];
                _mm256_store_si256((__m256i*)new_dist, v_new);

                for (int k = 0; k < 4; k++) {
                    if (du[k] < INF) {
                        int v = edges[j+k].v;
                        if (new_dist[k] < dist[v]) {
                            dist[v] = new_dist[k];
                            relaxed = true;
                        }
                    }
                }
            }

            // Scalar fallback za preostale grane u bloku
            for (; j < block_end; j++)
            {
                int u = edges[j].u;
                int v = edges[j].v;
                long wt = edges[j].weight;

                if (dist[u] < INF) {
                    long new_dist = dist[u] + wt;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        relaxed = true;
                    }
                }
            }
        }

        if (!relaxed) break;
    }

    return dist;
}

// ============================================================
//      AVX-512 VERZIJA SA CACHE OPTIMIZACIJOM
// ============================================================
std::vector<long> runBellmanFordSSSP_AVX512(
    int N, int E, int source_node_id,
    const std::vector<int>& src,
    const std::vector<int>& dst,
    const std::vector<int>& w)
{
    const long INF = std::numeric_limits<long>::max() / 2;

    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    struct EdgeTmp { int u, v, weight; };
    std::vector<EdgeTmp> edges(E);
    for (int i = 0; i < E; i++) {
        edges[i] = { src[i], dst[i], w[i] };
    }

    std::sort(edges.begin(), edges.end(), [](const EdgeTmp &a, const EdgeTmp &b){
        return a.u < b.u;
    });

    const int BLOCK_SIZE = 64;

    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;
        for (int block_start = 0; block_start < E; block_start += BLOCK_SIZE)
        {
            int block_end = std::min(block_start + BLOCK_SIZE, E);

            if (block_start + BLOCK_SIZE < E) {
                __builtin_prefetch(&edges[block_start + BLOCK_SIZE], 0, 3);
            }

            int j = block_start;

            // SIMD AVX-512 processing - 8 grana po iteraciji
            for (; j + 7 < block_end; j += 8)
            {
                alignas(64) long du[8] = { dist[edges[j+0].u], dist[edges[j+1].u],
                                            dist[edges[j+2].u], dist[edges[j+3].u],
                                            dist[edges[j+4].u], dist[edges[j+5].u],
                                            dist[edges[j+6].u], dist[edges[j+7].u] };
                alignas(64) long wt[8] = { (long)edges[j+0].weight, (long)edges[j+1].weight,
                                            (long)edges[j+2].weight, (long)edges[j+3].weight,
                                            (long)edges[j+4].weight, (long)edges[j+5].weight,
                                            (long)edges[j+6].weight, (long)edges[j+7].weight };

                __m512i v_du = _mm512_load_epi64(du);
                __m512i v_wt = _mm512_load_epi64(wt);
                __m512i v_new = _mm512_add_epi64(v_du, v_wt);

                alignas(64) long new_dist[8];
                _mm512_store_epi64(new_dist, v_new);

                for (int k = 0; k < 8; k++) {
                    if (du[k] < INF) {
                        int v = edges[j+k].v;
                        if (new_dist[k] < dist[v]) {
                            dist[v] = new_dist[k];
                            relaxed = true;
                        }
                    }
                }
            }

            // Scalar fallback za preostale grane u bloku
            for (; j < block_end; j++)
            {
                int u = edges[j].u;
                int v = edges[j].v;
                long wt = edges[j].weight;

                if (dist[u] < INF) {
                    long new_dist = dist[u] + wt;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        relaxed = true;
                    }
                }
            }
        }

        if (!relaxed) break;
    }

    return dist;
}
