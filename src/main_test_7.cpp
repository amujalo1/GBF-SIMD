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

    // ========== TEST 7: AVX-512 SoA VERZIJA ==========
    cout << "\n[TEST 7] AVX-512 SoA VERZIJA (Direktna SoA struktura)\n";
    cout << string(60, '-') << endl;
    
    // Ključni dio za VTune profilisanje
    auto start7 = chrono::high_resolution_clock::now();
    vector<long> distances7 = runBellmanFordSSSP_AVX512_SoA(gSoA, 0);
    auto end7 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed7 = end7 - start7;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed7.count() << " sekundi\n";
    if (distances7[last_node] >= numeric_limits<long>::max() / 2)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances7[last_node] << endl;

    // Čišćenje memorije
    delete[] g->edge;
    delete g;
    delete gSoA; 
    
    return 0;
}