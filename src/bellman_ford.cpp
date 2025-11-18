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
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;

    const long INF = std::numeric_limits<long>::max() / 2;
    std::vector<long> dist(no_of_nodes, INF);
    dist[source_node_id] = 0;

    // Pravimo privremeni vektor grana radi sortiranja
    std::vector<Edge> edges(graph->edge, graph->edge + no_of_edges);
    // sort 
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) {
                  return a.source < b.source;
              });

    const int BLOCK_SIZE = 64;

    for (int iter = 0; iter < no_of_nodes - 1; iter++)
    {
        bool relaxed = false;

        for (int block_start = 0; block_start < no_of_edges; block_start += BLOCK_SIZE)
        {
            int block_end = std::min(block_start + BLOCK_SIZE, no_of_edges);

            if (block_start + BLOCK_SIZE < no_of_edges)
                __builtin_prefetch(&edges[block_start + BLOCK_SIZE], 0, 3);

            int j = block_start;

            // --- SIMD: 4 grane odjednom ---
            for (; j + 3 < block_end; j += 4)
            {
                alignas(32) long du[4] = { dist[edges[j+0].source], dist[edges[j+1].source],
                                            dist[edges[j+2].source], dist[edges[j+3].source] };

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
                        int v = edges[j+k].destination;
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
                int u = edges[j].source;
                int v = edges[j].destination;
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

    // Pravimo privremeni vektor grana radi sortiranja
    std::vector<Edge> edges(graph->edge, graph->edge + E);
    // sort 
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) {
                  return a.source < b.source;
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
                    du[k] = dist[edges[j+k].source];
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
                        int v = edges[j+k].destination;
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
                int u = edges[j].source;
                int v = edges[j].destination;
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
//     AVX2 VERZIJA - DIREKTNA SoA (VEĆ SORTIRAN U UČITAVANJU)
// ============================================================
std::vector<long> runBellmanFordSSSP_AVX2_SoA(const GraphSoA* graph, int source_node_id)
{
    // Inicijalizacija
    int N = graph->num_nodes;
    int E = graph->num_edges;

    const long INF = std::numeric_limits<long>::max() / 2;
    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    // Referenciranje na VEĆ SORTIRANE podatke iz GraphSoA strukture
    const std::vector<int>& src = graph->sources;
    const std::vector<int>& dst = graph->destinations;
    const std::vector<int>& w = graph->weights;

    // Nema više EdgeSoA strukture, sortiranja i kopiranja!

    const int BLOCK_SIZE = 64; // Za prefetch

    // Glavna petlja Bellman-Forda
    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;

        for (int block_start = 0; block_start < E; block_start += BLOCK_SIZE)
        {
            int block_end = std::min(block_start + BLOCK_SIZE, E);

            // Prefetching sljedećeg bloka (Sada radi na samoj graph strukturi)
            if (block_start + BLOCK_SIZE < E) {
                __builtin_prefetch(&src[block_start + BLOCK_SIZE], 0, 3);
                __builtin_prefetch(&dst[block_start + BLOCK_SIZE], 0, 3);
                __builtin_prefetch(&w[block_start + BLOCK_SIZE], 0, 3);
            }

            int j = block_start;

            // --- SIMD: 4 grane odjednom (64-bit Long) ---
            for (; j + 3 < block_end; j += 4)
            {
                // Pripremamo ulazne podatke za AVX2 (4 x long = 256 bita)
                alignas(32) long du[4];
                alignas(32) long wt[4];

                for (int k = 0; k < 4; k++) {
                    // DIREKTAN PRISTUP SORTIRANIM VEKTORIMA
                    du[k] = dist[src[j+k]];
                    wt[k] = w[j+k];
                }

                // AVX2 relaksacija (64-bitno sabiranje)
                __m256i v_du = _mm256_load_si256((__m256i*)du);
                __m256i v_wt = _mm256_load_si256((__m256i*)wt);
                __m256i v_new = _mm256_add_epi64(v_du, v_wt);

                alignas(32) long new_dist[4];
                _mm256_store_si256((__m256i*)new_dist, v_new);

                // Skalarna provjera i ažuriranje
                for (int k = 0; k < 4; k++)
                {
                    if (du[k] < INF)
                    {
                        int v = dst[j+k];
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
                int u = src[j];
                int v = dst[j];
                long wt = w[j];

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
//     AVX-512 VERZIJA - DIREKTNA SoA (VEĆ SORTIRAN U UČITAVANJU)
// ============================================================
std::vector<long> runBellmanFordSSSP_AVX512_SoA(const GraphSoA* graph, int source_node_id)
{
    int N = graph->num_nodes;
    int E = graph->num_edges;

    // Distances
    const long INF = std::numeric_limits<long>::max() / 2;
    std::vector<long> dist(N, INF);
    dist[source_node_id] = 0;

    // Referenciranje na VEĆ SORTIRANE podatke
    const std::vector<int>& src = graph->sources;
    const std::vector<int>& dst = graph->destinations;
    const std::vector<int>& w = graph->weights;
    
    // Nema više EdgeSoA strukture i sortiranja/kopiranja!

    const int BLOCK_SIZE = 64;

    for (int iter = 0; iter < N - 1; iter++)
    {
        bool relaxed = false;

        for (int block_start = 0; block_start < E; block_start += BLOCK_SIZE)
        {
            int block_end = std::min(block_start + BLOCK_SIZE, E);

            // Prefetching sljedećeg bloka
            if (block_start + BLOCK_SIZE < E) {
                __builtin_prefetch(&src[block_start + BLOCK_SIZE], 0, 3);
                __builtin_prefetch(&dst[block_start + BLOCK_SIZE], 0, 3);
                __builtin_prefetch(&w[block_start + BLOCK_SIZE], 0, 3);
            }

            int j = block_start;

            // --- SIMD: 8 grana odjednom (64-bit) ---
            for (; j + 7 < block_end; j += 8)
            {
                // Pripremamo ulazne podatke za AVX-512
                alignas(64) long du[8];
                alignas(64) long wt[8];

                for (int k = 0; k < 8; k++) {
                    // DIREKTAN PRISTUP SORTIRANIM VEKTORIMA
                    du[k] = dist[src[j+k]];
                    wt[k] = w[j+k];
                }

                // AVX relaksacija
                __m512i v_du = _mm512_load_epi64(du);
                __m512i v_wt = _mm512_load_epi64(wt);
                __m512i v_new = _mm512_add_epi64(v_du, v_wt);

                alignas(64) long new_dist[8];
                _mm512_store_epi64(new_dist, v_new);

                // Skalarna provjera i ažuriranje (zbog zavisnosti i relaxed zastavice)
                for (int k = 0; k < 8; k++)
                {
                    if (du[k] < INF)
                    {
                        int v = dst[j+k]; // DIREKTAN PRISTUP
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
                int u = src[j];
                int v = dst[j];
                long wt = w[j];

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