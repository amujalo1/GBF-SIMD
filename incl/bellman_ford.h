#ifndef BELLAMAN_FORD_H
#define BELLAMAN_FORD_H

#include <vector>
#include "graph_utils.h"
#include <immintrin.h>

std::vector<long> runBellmanFordSSSP(Graph* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_CACHE(Graph* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_OMP(Graph* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_AVX2(Graph* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_AVX512(Graph* graph, int source_node_id);

// Nove SoA verzije koje direktno koriste GraphSoA strukturu
std::vector<long> runBellmanFordSSSP_AVX2_SoA(const GraphSoA* graph, int source_node_id);
std::vector<long> runBellmanFordSSSP_AVX512_SoA(const GraphSoA* graph, int source_node_id);

#endif