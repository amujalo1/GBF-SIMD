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

    // ========== TEST 2: CACHE VERZIJA ==========
    cout << "\n[TEST 2] CACHE OPTIMIZOVANA VERZIJA (CACHE + prefetch)\n";
    cout << string(60, '-') << endl;
    
    // Ključni dio za VTune profilisanje
    auto start2 = chrono::high_resolution_clock::now();
    vector<long> distances2 = runBellmanFordSSSP_CACHE(g, 0);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed2.count() << " sekundi\n";
    if (distances2[last_node] >= numeric_limits<int>::max() - 100)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances2[last_node] << endl;

    // Čišćenje memorije
    delete[] g->edge;
    delete g;
    delete gSoA; 
    
    return 0;
}