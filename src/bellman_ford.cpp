#include "bellman_ford.h"
#include <iostream>
#include <limits>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include <cstring>
#include <numeric>      
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
            long w = graph->edge[j].weight; 
            
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
std::vector<long> runBellmanFordSSSP_AVX2(Graph* graph, int source_node_id)
{
    int N = graph->num_nodes;
    int E = graph->num_edges;

    const long INF = std::numeric_limits<long>::max() / 2;
    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    // Pravimo privremeni vektor grana radi sortiranja
    struct EdgeTmp { int u, v, weight; };
    std::vector<EdgeTmp> edges(E);

    for (int i = 0; i < E; i++) {
        edges[i].u = graph->edge[i].source;
        edges[i].v = graph->edge[i].destination;
        edges[i].weight = graph->edge[i].weight;
    }

    // Sortiranje po source čvoru radi cache lokalnosti
    std::sort(edges.begin(), edges.end(), [](const EdgeTmp& a, const EdgeTmp& b){
        return a.u < b.u;
    });

    const int BLOCK_SIZE = 64;

    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;

        for (int block_start = 0; block_start < E; block_start += BLOCK_SIZE)
        {
            int block_end = std::min(block_start + BLOCK_SIZE, E);

            if (block_start + BLOCK_SIZE < E)
                __builtin_prefetch(&edges[block_start + BLOCK_SIZE], 0, 3);

            int j = block_start;

            // --- SIMD: 4 grane odjednom ---
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

                // scalar update
                for (int k = 0; k < 4; k++)
                {
                    if (du[k] < INF)
                    {
                        int v = edges[j+k].v;
                        if (new_dist[k] < dist[v])
                        {
                            dist[v] = new_dist[k];
                            relaxed = true;
                        }
                    }
                }
            }

            // --- Scalar ostatak ---
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
std::vector<long> runBellmanFordSSSP_AVX512(Graph* graph, int source_node_id)
{
    int N = graph->num_nodes;
    int E = graph->num_edges;

    const long INF = std::numeric_limits<long>::max() / 2;
    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    struct EdgeTmp { int u, v, weight; };
    std::vector<EdgeTmp> edges(E);

    for (int i = 0; i < E; i++) {
        edges[i].u = graph->edge[i].source;
        edges[i].v = graph->edge[i].destination;
        edges[i].weight = graph->edge[i].weight;
    }

    std::sort(edges.begin(), edges.end(), [](const EdgeTmp& a, const EdgeTmp& b){
        return a.u < b.u;
    });

    const int BLOCK_SIZE = 64;

    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;

        for (int block_start = 0; block_start < E; block_start += BLOCK_SIZE)
        {
            int block_end = std::min(block_start + BLOCK_SIZE, E);

            if (block_start + BLOCK_SIZE < E)
                __builtin_prefetch(&edges[block_start + BLOCK_SIZE], 0, 3);

            int j = block_start;

            // --- SIMD 8 grana odjednom ---
            for (; j + 7 < block_end; j += 8)
            {
                alignas(64) long du[8];
                alignas(64) long wt[8];

                for (int k = 0; k < 8; k++) {
                    du[k] = dist[edges[j+k].u];
                    wt[k] = edges[j+k].weight;
                }

                __m512i v_du = _mm512_load_epi64(du);
                __m512i v_wt = _mm512_load_epi64(wt);
                __m512i v_new = _mm512_add_epi64(v_du, v_wt);

                alignas(64) long new_dist[8];
                _mm512_store_epi64(new_dist, v_new);

                for (int k = 0; k < 8; k++)
                {
                    if (du[k] < INF)
                    {
                        int v = edges[j+k].v;
                        if (new_dist[k] < dist[v])
                        {
                            dist[v] = new_dist[k];
                            relaxed = true;
                        }
                    }
                }
            }

            // --- Scalar ostatak ---
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
//      AVX2 VERZIJA - SA ZAŠTITOM OD RACE CONDITION-A
// ============================================================
// std::vector<long> runBellmanFordSSSP_AVX2(
//     int N, int E, int source_node_id,
//     const std::vector<int>& src,
//     const std::vector<int>& dst,
//     const std::vector<int>& w)
// {
//     const long INF = std::numeric_limits<long>::max() / 2;
//     std::vector<long> dist(N, INF);
//     dist[source_node_id] = 0;

//     for (int iter = 0; iter < N - 1; iter++)
//     {
//         bool relaxed = false;
//         int e = 0;

//         AVX2: procesira 4 grane odjednom
//         for (; e + 3 < E; e += 4)
//         {
//             PROVJERA: Da li postoje konflikti u batch-u?
//             (dvije grane pišu u isti destination node)
//             bool conflict = (dst[e+0] == dst[e+1]) || (dst[e+0] == dst[e+2]) || (dst[e+0] == dst[e+3]) ||
//                            (dst[e+1] == dst[e+2]) || (dst[e+1] == dst[e+3]) ||
//                            (dst[e+2] == dst[e+3]);
            
//             if (conflict) {
//                 SCALAR PATH: procesiranje po jedna
//                 for (int k = 0; k < 4; k++) {
//                     if (dist[src[e+k]] < INF) {
//                         long new_dist = dist[src[e+k]] + w[e+k];
//                         if (new_dist < dist[dst[e+k]]) {
//                             dist[dst[e+k]] = new_dist;
//                             relaxed = true;
//                         }
//                     }
//                 }
//                 continue;
//             }

//             NEMA KONFLIKTA - Sigurno za AVX2
//             alignas(32) long du[4] = { 
//                 dist[src[e+0]], dist[src[e+1]],
//                 dist[src[e+2]], dist[src[e+3]]
//             };
            
//             alignas(32) long wt[4] = { 
//                 (long)w[e+0], (long)w[e+1],
//                 (long)w[e+2], (long)w[e+3]
//             };

//             __m256i v_du = _mm256_load_si256((__m256i*)du);
//             __m256i v_wt = _mm256_load_si256((__m256i*)wt);
//             __m256i v_new = _mm256_add_epi64(v_du, v_wt);

//             alignas(32) long new_dist[4];
//             _mm256_store_si256((__m256i*)new_dist, v_new);

//             Relaksacija - sigurna jer nema konflikata
//             for (int k = 0; k < 4; k++) {
//                 if (du[k] < INF && new_dist[k] < dist[dst[e+k]]) {
//                     dist[dst[e+k]] = new_dist[k];
//                     relaxed = true;
//                 }
//             }
//         }

//         Scalar fallback za preostale grane
//         for (; e < E; e++)
//         {
//             if (dist[src[e]] < INF) {
//                 long new_dist = dist[src[e]] + w[e];
//                 if (new_dist < dist[dst[e]]) {
//                     dist[dst[e]] = new_dist;
//                     relaxed = true;
//                 }
//             }
//         }

//         if (!relaxed) break;
//     }

//     return dist;
// }

// ============================================================
//      AVX-512 VERZIJA - SA ZAŠTITOM OD RACE CONDITION-A
// ============================================================
// std::vector<long> runBellmanFordSSSP_AVX512(
//     int N, int E, int source_node_id,
//     const std::vector<int>& src,
//     const std::vector<int>& dst,
//     const std::vector<int>& w)
// {
//     const long INF = std::numeric_limits<long>::max() / 2;
//     std::vector<long> dist(N, INF);
//     dist[source_node_id] = 0;

//     for (int iter = 0; iter < N - 1; iter++)
//     {
//         bool relaxed = false;
//         int e = 0;

//         for (; e + 7 < E; e += 8)
//         {
//             Provjera konflikata za 8 grana
//             bool conflict = false;
//             for (int i = 0; i < 8 && !conflict; i++) {
//                 for (int j = i + 1; j < 8; j++) {
//                     if (dst[e+i] == dst[e+j]) {
//                         conflict = true;
//                         break;
//                     }
//                 }
//             }
            
//             if (conflict) {
//                 Scalar path
//                 for (int k = 0; k < 8; k++) {
//                     if (dist[src[e+k]] < INF) {
//                         long new_dist = dist[src[e+k]] + w[e+k];
//                         if (new_dist < dist[dst[e+k]]) {
//                             dist[dst[e+k]] = new_dist;
//                             relaxed = true;
//                         }
//                     }
//                 }
//                 continue;
//             }

//             SIMD path
//             alignas(64) long du[8] = { 
//                 dist[src[e+0]], dist[src[e+1]],
//                 dist[src[e+2]], dist[src[e+3]],
//                 dist[src[e+4]], dist[src[e+5]],
//                 dist[src[e+6]], dist[src[e+7]]
//             };
            
//             alignas(64) long wt[8] = { 
//                 (long)w[e+0], (long)w[e+1],
//                 (long)w[e+2], (long)w[e+3],
//                 (long)w[e+4], (long)w[e+5],
//                 (long)w[e+6], (long)w[e+7]
//             };

//             __m512i v_du = _mm512_load_epi64(du);
//             __m512i v_wt = _mm512_load_epi64(wt);
//             __m512i v_new = _mm512_add_epi64(v_du, v_wt);

//             alignas(64) long new_dist[8];
//             _mm512_store_epi64(new_dist, v_new);

//             for (int k = 0; k < 8; k++) {
//                 if (du[k] < INF && new_dist[k] < dist[dst[e+k]]) {
//                     dist[dst[e+k]] = new_dist[k];
//                     relaxed = true;
//                 }
//             }
//         }

//         Scalar fallback
//         for (; e < E; e++)
//         {
//             if (dist[src[e]] < INF) {
//                 long new_dist = dist[src[e]] + w[e];
//                 if (new_dist < dist[dst[e]]) {
//                     dist[dst[e]] = new_dist;
//                     relaxed = true;
//                 }
//             }
//         }

//         if (!relaxed) break;
//     }

//     return dist;
// }

// ============================================================
//      ALTERNATIVA: VERZIJA SA ATOMICIMA (Thread-safe)
// ============================================================
// std::vector<long> runBellmanFordSSSP_AVX2_Atomic(
//     int N, int E, int source_node_id,
//     const std::vector<int>& src,
//     const std::vector<int>& dst,
//     const std::vector<int>& w)
// {
//     const long INF = std::numeric_limits<long>::max() / 2;
//     std::vector<std::atomic<long>> dist_atomic(N);
    
//     for (int i = 0; i < N; i++) {
//         dist_atomic[i].store(INF, std::memory_order_relaxed);
//     }
//     dist_atomic[source_node_id].store(0, std::memory_order_relaxed);

//     for (int iter = 0; iter < N - 1; iter++)
//     {
//         bool relaxed = false;
//         int e = 0;

//         for (; e + 3 < E; e += 4)
//         {
//             alignas(32) long du[4] = { 
//                 dist_atomic[src[e+0]].load(std::memory_order_relaxed),
//                 dist_atomic[src[e+1]].load(std::memory_order_relaxed),
//                 dist_atomic[src[e+2]].load(std::memory_order_relaxed),
//                 dist_atomic[src[e+3]].load(std::memory_order_relaxed)
//             };
            
//             alignas(32) long wt[4] = { 
//                 (long)w[e+0], (long)w[e+1],
//                 (long)w[e+2], (long)w[e+3]
//             };

//             __m256i v_du = _mm256_load_si256((__m256i*)du);
//             __m256i v_wt = _mm256_load_si256((__m256i*)wt);
//             __m256i v_new = _mm256_add_epi64(v_du, v_wt);

//             alignas(32) long new_dist[4];
//             _mm256_store_si256((__m256i*)new_dist, v_new);

//             Atomic compare-and-swap relaksacija
//             for (int k = 0; k < 4; k++) {
//                 if (du[k] < INF) {
//                     int v = dst[e+k];
//                     long old_dist = dist_atomic[v].load(std::memory_order_relaxed);
//                     while (new_dist[k] < old_dist) {
//                         if (dist_atomic[v].compare_exchange_weak(old_dist, new_dist[k],
//                                                                  std::memory_order_relaxed)) {
//                             relaxed = true;
//                             break;
//                         }
//                     }
//                 }
//             }
//         }

//         for (; e < E; e++)
//         {
//             long du = dist_atomic[src[e]].load(std::memory_order_relaxed);
//             if (du < INF) {
//                 long new_dist = du + w[e];
//                 int v = dst[e];
//                 long old_dist = dist_atomic[v].load(std::memory_order_relaxed);
//                 while (new_dist < old_dist) {
//                     if (dist_atomic[v].compare_exchange_weak(old_dist, new_dist,
//                                                              std::memory_order_relaxed)) {
//                         relaxed = true;
//                         break;
//                     }
//                 }
//             }
//         }

//         if (!relaxed) break;
//     }

//     Konvertuj atomic nazad u obični vektor
//     std::vector<long> dist(N);
//     for (int i = 0; i < N; i++) {
//         dist[i] = dist_atomic[i].load(std::memory_order_relaxed);
//     }
//     return dist;
// }