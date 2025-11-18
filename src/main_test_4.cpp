#include "graph_utils.h"
#include "bellman_ford.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <limits>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

int main() {
    // --- DIJELJENA LOGIKA ZA INICIJALIZACIJU (Kopirati iz main_test_1.cpp) ---
    string folder = "graph";
    string txtFile = folder + "/graf.txt";
    if (!fs::exists(folder)) { fs::create_directory(folder); }
    if (!fs::exists(txtFile)) {
        createGraph(txtFile, 200000, 8000000, -15, 35);
    }
    Graph* g = readGraph(txtFile);
    GraphSoA* gSoA = readGraphSoA(txtFile); 
    if (!g || !gSoA) return 1;

    cout << "[INFO] Ucitano: " << g->num_nodes << " cvorova, "
         << g->num_edges << " grana.\n";
    cout << string(60, '=') << endl;
    int last_node = g->num_nodes - 1;
    // --- KRAJ DIJELJENE LOGIKE ---

    // ========== TEST 4: AVX2 SIMD VERZIJA (AoS) ==========
    cout << "\n[TEST 4] AVX2 SIMD VERZIJA (AoS + SIMD relaksacija)\n";
    cout << string(60, '-') << endl;
    
    // Ključni dio za VTune profilisanje
    auto start4 = chrono::high_resolution_clock::now();
    vector<long> distances4 = runBellmanFordSSSP_AVX2(g, 0);
    auto end4 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed4 = end4 - start4;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed4.count() << " sekundi\n";
    if (distances4[last_node] >= numeric_limits<int>::max() - 100)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances4[last_node] << endl;

    // Čišćenje memorije
    delete[] g->edge;
    delete g;
    delete gSoA; 
    
    return 0;
}