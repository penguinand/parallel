#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <queue>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include <arm_neon.h> // 包含NEON头文件

using namespace hnswlib;

// 搜索结果结构体
struct SearchResult {
    float recall;
    int64_t latency; // 单位微秒 (μs)
};

// 数据加载函数
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

float inner_product_neon(const float* base, const float* query, size_t vecdim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (size_t d = 0; d < vecdim; d += 4) {
        float32x4_t b = vld1q_f32(base + d);
        float32x4_t q = vld1q_f32(query + d);
        sum = vmlaq_f32(sum, b, q);
    }

    // 水平求和
    float32x2_t sum_low = vget_low_f32(sum);
    float32x2_t sum_high = vget_high_f32(sum);
    float32x2_t temp_sum = vpadd_f32(sum_low, sum_high);
    float32x2_t final_sum = vpadd_f32(temp_sum, temp_sum);

    return vget_lane_f32(final_sum, 0);
}

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < base_number; ++i) {
        float dis = 1 - inner_product_neon(base + i * vecdim, query, vecdim);

        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}

// 主函数
int main() {
    // 加载数据
    size_t test_number = 0, base_number=0, vecdim=0, test_gt_d=0;
    std::string data_path = "anndata";
    
    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // 仅测试前 2000 条查询
    test_number = 2000;
    const size_t k = 10;
    std::vector<SearchResult> results(test_number);
	
    build_index(base, base_number, vecdim);

    // 遍历所有查询
    for (size_t i = 0; i < test_number; ++i) {
        // 计时开始
        auto start = std::chrono::high_resolution_clock::now();

        // 执行暴力搜索
        auto res = flat_search_neon(base, test_query + i * vecdim, base_number, vecdim, k);

        // 计算耗时（微秒）
        auto end = std::chrono::high_resolution_clock::now();
        int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 计算召回率
        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k; ++j) {
            gt_set.insert(test_gt[i * test_gt_d + j]);
        }
        size_t correct = 0;
        while (!res.empty()) {
            if (gt_set.count(res.top().second)) correct++;
            res.pop();
        }
        float recall = static_cast<float>(correct) / k;

        // 保存结果
        results[i] = {recall, latency};
    }

    // 统计平均结果
    float avg_recall = 0.0f;
    int64_t avg_latency = 0;
    for (const auto& res : results) {
        avg_recall += res.recall;
        avg_latency += res.latency;
    }
    avg_recall /= test_number;
    avg_latency /= test_number;

    std::cout << "Average Recall: " << avg_recall << std::endl;
    std::cout << "Average Latency: " << avg_latency << " μs" << std::endl;

    // 释放内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}
