#ifndef BELLAMAN_FORD_H
#define BELLAMAN_FORD_H

#include <vector>
#include "graph_utils.h"
#include <immintrin.h>

std::vector<long> runBellmanFordSSSP(Graph* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_CACHE(Graph* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_OMP(Graph* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_AVX2(
    int N, int E, int source_node_id,
    const std::vector<int>& src,
    const std::vector<int>& dst,
    const std::vector<int>& w);
std::vector<long> runBellmanFordSSSP_AVX512(
    int N, int E, int source_node_id,
    const std::vector<int>& src,
    const std::vector<int>& dst,
    const std::vector<int>& w);
#endif
