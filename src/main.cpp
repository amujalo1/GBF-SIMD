#include "graph_utils.h"
#include "bellman_ford.h"
#include <iostream>
#include <fstream>
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
    
    // Učitavanje grafa u SoA format
    GraphSoA* gSoA = readGraphSoA(txtFile);
    if (!gSoA) {
        cerr << "[ERROR] Ne mogu ucitati graf (SoA)!" << endl;
        delete[] g->edge;
        delete g;
        return 1;
    }
    
    cout << "[INFO] Ucitano: " << g->num_nodes << " cvorova, "
         << g->num_edges << " grana.\n";
    cout << "[INFO] SoA format: " << gSoA->num_nodes << " cvorova, "
         << gSoA->num_edges << " grana.\n";
    cout << string(60, '=') << endl;
    
    int last_node = g->num_nodes - 1;

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

    // ========== TEST 4: AVX2 SIMD VERZIJA (AoS) ==========
    cout << "\n[TEST 4] AVX2 SIMD VERZIJA (AoS + SIMD relaksacija)\n";
    cout << string(60, '-') << endl;

    auto start4 = chrono::high_resolution_clock::now();
    vector<long> distances4 = runBellmanFordSSSP_AVX2(g, 0);
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

    // ========== TEST 5: AVX-512 SIMD VERZIJA (AoS) ==========
    cout << "\n[TEST 5] AVX-512 SIMD VERZIJA (AoS + 64-bit SIMD relaksacija)\n";
    cout << string(60, '-') << endl;

    auto start5 = chrono::high_resolution_clock::now();
    vector<long> distances5 = runBellmanFordSSSP_AVX512(g, 0);
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

    // ========== TEST 6: AVX2 SoA VERZIJA ==========
    cout << "\n[TEST 6] AVX2 SoA VERZIJA (Direktna SoA struktura)\n";
    cout << string(60, '-') << endl;

    auto start6 = chrono::high_resolution_clock::now();
    vector<long> distances6 = runBellmanFordSSSP_AVX2_SoA(gSoA, 0);
    auto end6 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed6 = end6 - start6;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed6.count() << " sekundi\n";
    if (distances6[last_node] >= numeric_limits<long>::max() / 2)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances6[last_node] << endl;

    double speedup6 = elapsed1.count() / elapsed6.count();
    cout << "[UBRZANJE] " << fixed << setprecision(2) << speedup6 << "x " 
         << (speedup6 > 1.0 ? "(BRZE)" : "(SPORIJE)") << endl;
    cout << "[KOREKTNOST] " << (distances1 == distances6 ? "✓ TACNO" : "✗ NETACNO") << endl;

    // ========== TEST 7: AVX-512 SoA VERZIJA ==========
    cout << "\n[TEST 7] AVX-512 SoA VERZIJA (Direktna SoA struktura)\n";
    cout << string(60, '-') << endl;

    auto start7 = chrono::high_resolution_clock::now();
    vector<long> distances7 = runBellmanFordSSSP_AVX512_SoA(gSoA, 0);
    auto end7 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed7 = end7 - start7;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed7.count() << " sekundi\n";
    if (distances7[last_node] >= numeric_limits<long>::max() / 2)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan iz izvora.\n";
    else
        cout << "[REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances7[last_node] << endl;

    double speedup7 = elapsed1.count() / elapsed7.count();
    cout << "[UBRZANJE] " << fixed << setprecision(2) << speedup7 << "x " 
         << (speedup7 > 1.0 ? "(BRZE)" : "(SPORIJE)") << endl;
    cout << "[KOREKTNOST] " << (distances1 == distances7 ? "✓ TACNO" : "✗ NETACNO") << endl;

    // ==================== FINALNI REZIME ====================
    cout << "\n" << string(60, '=') << endl;
    cout << "REZIME PERFORMANSI:\n";
    cout << string(60, '=') << endl;

    cout << fixed << setprecision(3);
    cout << "Originalna:    " << setw(8) << elapsed1.count() << "s  (baseline)\n";
    cout << "CACHE:         " << setw(8) << elapsed2.count() << "s  (" << setprecision(2) << speedup2 << "x)\n";
    cout << "OpenMP:        " << setw(8) << elapsed3.count() << "s  (" << setprecision(2) << speedup3 << "x)\n";
    cout << "AVX2 (AoS):    " << setw(8) << elapsed4.count() << "s  (" << setprecision(2) << speedup4 << "x)\n";
    cout << "AVX-512 (AoS): " << setw(8) << elapsed5.count() << "s  (" << setprecision(2) << speedup5 << "x)\n";
    cout << "AVX2 (SoA):    " << setw(8) << elapsed6.count() << "s  (" << setprecision(2) << speedup6 << "x)\n";
    cout << "AVX-512 (SoA): " << setw(8) << elapsed7.count() << "s  (" << setprecision(2) << speedup7 << "x)\n";

    double best_time = min({elapsed1.count(), elapsed2.count(), elapsed3.count(), 
                            elapsed4.count(), elapsed5.count(), elapsed6.count(), elapsed7.count()});
    string best_version;
    if (best_time == elapsed1.count()) best_version = "Originalna";
    else if (best_time == elapsed2.count()) best_version = "CACHE";
    else if (best_time == elapsed3.count()) best_version = "OpenMP";
    else if (best_time == elapsed4.count()) best_version = "AVX2 (AoS)";
    else if (best_time == elapsed5.count()) best_version = "AVX-512 (AoS)";
    else if (best_time == elapsed6.count()) best_version = "AVX2 (SoA)";
    else best_version = "AVX-512 (SoA)";

    cout << "\n[POBJEDNIK] " << best_version << " verzija je NAJBRZA!\n";
    
    cout << "\n[ANALIZA SoA vs AoS]\n";
    cout << string(60, '-') << endl;
    double aos_avx2_time = elapsed4.count();
    double soa_avx2_time = elapsed6.count();
    double aos_avx512_time = elapsed5.count();
    double soa_avx512_time = elapsed7.count();
    
    double soa_avx2_improvement = (aos_avx2_time / soa_avx2_time);
    double soa_avx512_improvement = (aos_avx512_time / soa_avx512_time);
    
    cout << "AVX2:    SoA je " << fixed << setprecision(2) 
         << (soa_avx2_improvement > 1.0 ? 
             to_string(soa_avx2_improvement) + "x BRZI" : 
             to_string(1.0/soa_avx2_improvement) + "x SPORIJI") << " od AoS\n";
    cout << "AVX-512: SoA je " << fixed << setprecision(2) 
         << (soa_avx512_improvement > 1.0 ? 
             to_string(soa_avx512_improvement) + "x BRZI" : 
             to_string(1.0/soa_avx512_improvement) + "x SPORIJI") << " od AoS\n";
    
    cout << string(60, '=') << endl;

    // Čišćenje memorije
    delete[] g->edge;
    delete g;
    delete gSoA;
    
    return 0;
}