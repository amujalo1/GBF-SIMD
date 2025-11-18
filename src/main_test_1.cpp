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

// Uključi strukture za Graph i GraphSoA i funkcije za Bellman-Ford

int main() {
    // --- DIJELJENA LOGIKA ZA INICIJALIZACIJU (Ponovljeno u svakom fajlu) ---
    string folder = "graph";
    string txtFile = folder + "/graf.txt";

    // Kreiranje foldera
    if (!fs::exists(folder)) {
        fs::create_directory(folder);
        cout << "[INFO] Kreiran folder: " << folder << endl;
    }

    // Kreiranje grafa
    if (!fs::exists(txtFile)) {
        cout << "[INFO] Fajl ne postoji, generisem graf..." << endl;
        bool ok = createGraph(txtFile, 200000, 8000000, -15, 35);
        if (!ok) {
            cerr << "[ERROR] Neuspjesno generisanje grafa!" << endl;
            return 1;
        }
    } else {
        cout << "[INFO] Graf vec postoji → preskacem generisanje." << endl;
    }

    // Učitavanje grafa za originalne verzije (AoS format)
    Graph* g = readGraph(txtFile);
    if (!g) {
        cerr << "[ERROR] Ne mogu ucitati graf (AoS)!" << endl;
        return 1;
    }
    
    // Učitavanje grafa u SoA format (Potrebno za testove 6 i 7, ovdje samo za simetričnost čišćenja)
    GraphSoA* gSoA = readGraphSoA(txtFile);
    if (!gSoA) {
        cerr << "[ERROR] Ne mogu ucitati graf (SoA)!" << endl;
        delete[] g->edge;
        delete g;
        return 1;
    }

    cout << "[INFO] Ucitano: " << g->num_nodes << " cvorova, "
         << g->num_edges << " grana.\n";
    cout << string(60, '=') << endl;
    
    int last_node = g->num_nodes - 1;
    // --- KRAJ DIJELJENE LOGIKE ---

    // ========== TEST 1: ORIGINALNA VERZIJA ==========
    cout << "\n[TEST 1] ORIGINALNA VERZIJA (standardna implementacija)\n";
    cout << string(60, '-') << endl;
    
    // Ključni dio za VTune profilisanje
    auto start1 = chrono::high_resolution_clock::now();
    vector<long> distances1 = runBellmanFordSSSP(g, 0);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed1 = end1 - start1;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed1.count() << " sekundi\n";
    if (distances1[last_node] >= numeric_limits<int>::max() - 100)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances1[last_node] << endl;
    
    // Čišćenje memorije
    delete[] g->edge;
    delete g;
    delete gSoA; 
    
    return 0;
}