#include "graph_utils.h"
#include "bellman_ford.h"
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

int main() {
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
        bool ok = createGraph(txtFile, 50000, 3000000, -15, 25);
        if (!ok) {
            cerr << "[ERROR] Neuspjesno generisanje grafa!" << endl;
            return 1;
        }
    } else {
        cout << "[INFO] Graf vec postoji → preskacem generisanje." << endl;
    }
    
    // Učitavanje grafa
    Graph* g = readGraph(txtFile);
    if (!g) {
        cerr << "[ERROR] Ne mogu ucitati graf!" << endl;
        return 1;
    }
    
    cout << "[INFO] Ucitano: " << g->num_nodes << " cvorova, "
         << g->num_edges << " grana.\n";
    cout << string(60, '=') << endl;
    
    int last_node = g->num_nodes - 1;

    // ==================== Konverzija AoS → SoA ====================
    vector<int> src(g->num_edges);
    vector<int> dst(g->num_edges);
    vector<int> w(g->num_edges);
    for (int i = 0; i < g->num_edges; i++) {
        src[i] = g->edge[i].source;
        dst[i] = g->edge[i].destination;
        w[i]   = g->edge[i].weight;
    }

    // ========== TEST 1: ORIGINALNA VERZIJA ==========
    cout << "\n[TEST 1] ORIGINALNA VERZIJA (standardna implementacija)\n";
    cout << string(60, '-') << endl;
    
    auto start1 = chrono::high_resolution_clock::now();
    vector<long> distances1 = runBellmanFordSSSP(g, 0);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed1 = end1 - start1;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed1.count() << " sekundi\n";
    if (distances1[last_node] >= numeric_limits<int>::max() - 100)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances1[last_node] << endl;

    // ========== TEST 2: CACHE VERZIJA ==========
    cout << "\n[TEST 2] CACHE OPTIMIZOVANA VERZIJA (CACHE + prefetch)\n";
    cout << string(60, '-') << endl;
    
    auto start2 = chrono::high_resolution_clock::now();
    vector<long> distances2 = runBellmanFordSSSP_CACHE(g, 0);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed2.count() << " sekundi\n";
    if (distances2[last_node] >= numeric_limits<int>::max() - 100)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances2[last_node] << endl;

    double speedup2 = elapsed1.count() / elapsed2.count();
    cout << "[UBRZANJE] " << fixed << setprecision(2) << speedup2 << "x " 
         << (speedup2 > 1.0 ? "(BRZE)" : "(SPORIJE)") << endl;
    cout << "[KOREKTNOST] " << (distances1 == distances2 ? "✓ TACNO" : "✗ NETACNO") << endl;

    // ========== TEST 3: OpenMP VERZIJA ==========
    cout << "\n[TEST 3] OpenMP PARALELIZOVANA VERZIJA (multi-threading)\n";
    cout << string(60, '-') << endl;
    
    auto start3 = chrono::high_resolution_clock::now();
    vector<long> distances3 = runBellmanFordSSSP_OMP(g, 0);
    auto end3 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed3 = end3 - start3;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed3.count() << " sekundi\n";
    if (distances3[last_node] >= numeric_limits<int>::max() - 100)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances3[last_node] << endl;

    double speedup3 = elapsed1.count() / elapsed3.count();
    cout << "[UBRZANJE] " << fixed << setprecision(2) << speedup3 << "x " 
         << (speedup3 > 1.0 ? "(BRZE)" : "(SPORIJE)") << endl;
    cout << "[KOREKTNOST] " << (distances1 == distances3 ? "✓ TACNO" : "✗ NETACNO") << endl;

    // ========== TEST 4: AVX2 SIMD VERZIJA ==========
    cout << "\n[TEST 4] AVX2 SIMD VERZIJA (SoA + SIMD relaksacija)\n";
    cout << string(60, '-') << endl;

    auto start4 = chrono::high_resolution_clock::now();
    vector<long> distances4 = runBellmanFordSSSP_AVX2(g->num_nodes, g->num_edges, 0, src, dst, w);
    auto end4 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed4 = end4 - start4;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed4.count() << " sekundi\n";
    if (distances4[last_node] >= numeric_limits<int>::max() - 100)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances4[last_node] << endl;

    double speedup4 = elapsed1.count() / elapsed4.count();
    cout << "[UBRZANJE] " << fixed << setprecision(2) << speedup4 << "x " 
         << (speedup4 > 1.0 ? "(BRZE)" : "(SPORIJE)") << endl;
    cout << "[KOREKTNOST] " << (distances1 == distances4 ? "✓ TACNO" : "✗ NETACNO") << endl;

    // ========== TEST 5: AVX-512 SIMD VERZIJA ==========
    cout << "\n[TEST 5] AVX-512 SIMD VERZIJA (SoA + 64-bit SIMD relaksacija)\n";
    cout << string(60, '-') << endl;

    auto start5 = chrono::high_resolution_clock::now();
    vector<long> distances5 = runBellmanFordSSSP_AVX512(g->num_nodes, g->num_edges, 0, src, dst, w);
    auto end5 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed5 = end5 - start5;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed5.count() << " sekundi\n";
    if (distances5[last_node] >= numeric_limits<long>::max() / 2)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances5[last_node] << endl;

    double speedup5 = elapsed1.count() / elapsed5.count();
    cout << "[UBRZANJE] " << fixed << setprecision(2) << speedup5 << "x " 
         << (speedup5 > 1.0 ? "(BRZE)" : "(SPORIJE)") << endl;
    cout << "[KOREKTNOST] " << (distances1 == distances5 ? "✓ TACNO" : "✗ NETACNO") << endl;

    // ==================== FINALNI REZIME ====================
    cout << "\n" << string(60, '=') << endl;
    cout << "REZIME PERFORMANSI:\n";
    cout << string(60, '=') << endl;

    cout << fixed << setprecision(3);
    cout << "Originalna:    " << setw(8) << elapsed1.count() << "s  (baseline)\n";
    cout << "CACHE:          " << setw(8) << elapsed2.count() << "s  (" << setprecision(2) << speedup2 << "x)\n";
    cout << "OpenMP:        " << setw(8) << elapsed3.count() << "s  (" << setprecision(2) << speedup3 << "x)\n";
    cout << "AVX2:          " << setw(8) << elapsed4.count() << "s  (" << setprecision(2) << speedup4 << "x)\n";
    cout << "AVX-512:       " << setw(8) << elapsed5.count() << "s  (" << setprecision(2) << speedup5 << "x)\n";

    double best_time = min({elapsed1.count(), elapsed2.count(), elapsed3.count(), elapsed4.count(), elapsed5.count()});
    string best_version;
    if (best_time == elapsed1.count()) best_version = "Originalna";
    else if (best_time == elapsed2.count()) best_version = "CACHE";
    else if (best_time == elapsed3.count()) best_version = "OpenMP";
    else if (best_time == elapsed4.count()) best_version = "AVX2";
    else best_version = "AVX-512";

    cout << "\n[POBJEDNIK] " << best_version << " verzija je NAJBRZA!\n";
    cout << string(60, '=') << endl;

    delete[] g->edge;
    delete g;
    return 0;
}
