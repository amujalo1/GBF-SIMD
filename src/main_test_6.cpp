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

    // ========== TEST 6: AVX2 SoA VERZIJA ==========
    cout << "\n[TEST 6] AVX2 SoA VERZIJA (Direktna SoA struktura)\n";
    cout << string(60, '-') << endl;
    
    // Ključni dio za VTune profilisanje
    auto start6 = chrono::high_resolution_clock::now();
    vector<long> distances6 = runBellmanFordSSSP_AVX2_SoA(gSoA, 0);
    auto end6 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed6 = end6 - start6;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed6.count() << " sekundi\n";
    if (distances6[last_node] >= numeric_limits<long>::max() / 2)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances6[last_node] << endl;

    // Čišćenje memorije
    delete[] g->edge;
    delete g;
    delete gSoA; 
    
    return 0;
}