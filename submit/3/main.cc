#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <omp.h>
#include "scan_0.93_810.h"
#include <arm_neon.h>

using namespace std;

template<typename T>
T *LoadData(const string& path, size_t& n, size_t& d) {
    ifstream fin(path, ios::binary);
    if (!fin) throw runtime_error("cannot open " + path);
    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    T* data = new T[n * d];
    fin.read(reinterpret_cast<char*>(data), n * d * sizeof(T));
    fin.close();
    cerr << "load data " << path
         << "  #=" << n
         << "  dim=" << d
         << " sizeof(T)=" << sizeof(T)
         << "\n";
    return data;
}

int main() {
    const string prefix = "/anndata/";
    size_t nq = 0, nb = 0, d_q = 0, d_gt = 0;
    float* queries = LoadData<float>(prefix + "DEEP100K.query.fbin", nq, d_q);
    int*   gt_data = LoadData<int>(
        prefix + "DEEP100K.gt.query.100k.top100.bin", nq, d_gt);
    float* base    = LoadData<float>(prefix + "DEEP100K.base.100k.fbin", nb, d_q);

    size_t Q = min<size_t>(2000, nq);
    size_t k = 10;

    PQIndex index(d_q);
    index.train(base, nb);
    index.encode(base, nb);

    vector<vector<pair<float,uint32_t>>> results;
    auto t0 = chrono::high_resolution_clock::now();
    batch_search(index, queries, Q, k, results);
    auto t1 = chrono::high_resolution_clock::now();

    double total_us = chrono::duration<double, micro>(t1 - t0).count();
    double avg_lat  = total_us / Q;

    double sum_rec = 0.0;
    for (size_t i = 0; i < Q; ++i) {
        set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(uint32_t(gt_data[i * d_gt + j]));
        }
        size_t cnt = 0;
        for (auto &pr : results[i]) {
            if (gtset.count(pr.second)) ++cnt;
        }
        sum_rec += double(cnt) / k;
    }
    double avg_rec = sum_rec / Q;

    cout << "Average Recall:  " << avg_rec << "\n";
    cout << "Average Latency: " << avg_lat << " μs\n";

    delete[] queries;
    delete[] gt_data;
    delete[] base;
    return 0;
}