#include <hnswlib/hnswlib.h>

class PQHNSWIndex {
private:
    PQIndex pq_index;
    hnswlib::HierarchicalNSW<float>* hnsw_index;
    size_t dim;

public:
    PQHNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200)
        : pq_index(dim), dim(dim)
    {
        hnsw_index = new hnswlib::HierarchicalNSW<float>(
            new hnswlib::L2Space(dim), 1 << 24, M, ef_construction);
    }

    ~PQHNSWIndex() {
        delete hnsw_index->space_;
        delete hnsw_index;
    }

    void train(const float* data, size_t n) {
        // 先训练PQ编码器
        pq_index.train(data, n);

        // 使用PQ编码构建HNSW图
        std::vector<float> encoded_data(n * dim);
        pq_index.encode(data, n);

        // 构建HNSW索引
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            hnsw_index->addPoint(encoded_data.data() + i * dim, i);
        }
    }

    std::vector<uint32_t> search(const float* query, size_t k, size_t ef_search = 100) {
        // 获取HNSW的候选集
        auto candidates = hnsw_index->searchKnn(query, ef_search);

        // 使用PQ进行精确距离计算
        thread_local std::vector<float> lut(SUBSPACE_NUM * CENTROID_NUM);
        pq_index.precompute_LUT(query, lut.data());
        const uint8_t* codes = pq_index.get_codes();

        std::priority_queue<std::pair<float, uint32_t>> pq;
        for (auto& cand : candidates) {
            float dis = 0.0f;
            const uint8_t* code = codes + cand.second * SUBSPACE_NUM;
            for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
                dis += lut[s * CENTROID_NUM + code[s]];
            }
            pq.emplace(dis, cand.second);
            if (pq.size() > k) pq.pop();
        }

        // 收集结果
        std::vector<uint32_t> result;
        while (!pq.empty()) {
            result.push_back(pq.top().second);
            pq.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    void encode(float* data, size_t n) {
        pq_index.encode(data, n);
    }
};