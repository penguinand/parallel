#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <queue>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "scan_0.9591_5411.h"
#include <arm_neon.h>

using namespace hnswlib;

struct SearchResult {
    float   recall;
    int64_t latency;
};

template<typename T>
T *LoadData(const std::string& path, size_t& n, size_t& d) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("cannot open " + path);
    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    T* data = new T[n * d];
    fin.read(reinterpret_cast<char*>(data), n * d * sizeof(T));
    fin.close();
    std::cerr << "load data " << path
              << "  #=" << n
              << "  dim=" << d
              << "  sizeof(T)=" << sizeof(T)
              << "\n";
    return data;
}

int main() {
    size_t test_number = 0, base_number = 0;
    size_t vecdim     = 0, test_gt_d = 0;
    const std::string data_path = "/anndata/";

    float* test_query = LoadData<float>(
        data_path + "DEEP100K.query.fbin",
        test_number, vecdim);
    int*   test_gt    = LoadData<int>(
        data_path + "DEEP100K.gt.query.100k.top100.bin",
        test_number, test_gt_d);
    float* base       = LoadData<float>(
        data_path + "DEEP100K.base.100k.fbin",
        base_number, vecdim);

    test_number = std::min<size_t>(2000, test_number);
    const size_t k = 10;
    std::vector<SearchResult> results(test_number);

    PQIndex pq_index(vecdim);
    pq_index.train(base, base_number);
    pq_index.encode(base, base_number);

    for (size_t i = 0; i < test_number; ++i) {
        const float* qptr = test_query + i * vecdim;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = pq_search(pq_index, qptr, base_number, k);
        auto t1 = std::chrono::high_resolution_clock::now();

        int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(
                              t1 - t0
                          ).count();

        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k; ++j) {
            gt_set.insert(static_cast<uint32_t>(
                              test_gt[i * test_gt_d + j]
                          ));
        }
        size_t correct = 0;
        while (!res.empty()) {
            if (gt_set.count(res.top().second)) ++correct;
            res.pop();
        }
        float recall = float(correct) / k;
        results[i] = {recall, latency};
    }

    double avg_recall = 0;
    int64_t avg_lat    = 0;
    for (auto &r : results) {
        avg_recall += r.recall;
        avg_lat    += r.latency;
    }
    avg_recall /= results.size();
    avg_lat    /= results.size();

    std::cout << "Average Recall:  " << avg_recall << std::endl;
    std::cout << "Average Latency: " << avg_lat << " μs" << std::endl;

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    return 0;
}