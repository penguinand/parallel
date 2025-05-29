#include <vector>
#include <cstring>
#include <random>
#include <numeric>
#include <limits>
#include <cfloat>          // FLT_MAX
#include <queue>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <arm_neon.h>
#include "hnswlib/hnswlib/hnswlib.h"

// ==== 参数调优（极限模式） ====
// coarse‐quantizer
static constexpr size_t DIM             = 128;
static constexpr size_t N_CLUSTERS      = 100;
static constexpr size_t IVF_KMEANS_ITER = 10;
static constexpr size_t N_PROBE         = 20;    // 探更大范围

// PQ
static constexpr size_t SUBSPACE_NUM  = 16;
static constexpr size_t SUBSPACE_DIM  = DIM / SUBSPACE_NUM;  // =8
static constexpr size_t CENTROID_NUM  = 256;
static constexpr size_t KMEANS_ITER   = 10;

// HNSW per‐cluster
static constexpr int HNSW_M           = 64;   // build时每点出度
static constexpr int HNSW_EFC         = 800;  // efConstruction
static constexpr int HNSW_EFSEARCH    = 500;  // efSearch during query

// Re‐rank 参数
static constexpr size_t RERANK_FACTOR = 10;    // top‐L = K * RERANK_FACTOR

// ==== thread‐local storage for Asymmetric‐PQ ====
static thread_local const float* tls_lut_ptr       = nullptr;
static thread_local bool        tls_use_asymmetric = false;

// ==== IVFIndex ====
class IVFIndex {
private:
    size_t dim_;
    std::vector<float>                  centroids_;  // N_CLUSTERS * dim_
    std::vector<std::vector<uint32_t>>  clusters_;   // 每个 cluster 的点索引

    void kmeans_init(const float* data, size_t n) {
        // 随机挑 N_CLUSTERS 个初始质心（KMeans++ 也可）
        std::mt19937 rng;
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        centroids_.resize(N_CLUSTERS * dim_);
        for (size_t c = 0; c < N_CLUSTERS; ++c) {
            std::memcpy(&centroids_[c*dim_],
                        data + idx[c]*dim_,
                        dim_ * sizeof(float));
        }
    }

public:
    IVFIndex(size_t d): dim_(d) {}

    // 训练 coarse‐quantizer
    void build(const float* data, size_t n) {
        kmeans_init(data, n);
        std::vector<size_t> assign(n);
        for (size_t it = 0; it < IVF_KMEANS_ITER; ++it) {
            clusters_.assign(N_CLUSTERS, {});
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float* vec = data + i*dim_;
                float best = FLT_MAX;
                size_t bid = 0;
                for (size_t c = 0; c < N_CLUSTERS; ++c) {
                    const float* cent = &centroids_[c*dim_];
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    size_t d = 0;
                    for (; d+4 <= dim_; d+=4) {
                        auto v  = vld1q_f32(vec + d);
                        auto cb = vld1q_f32(cent + d);
                        auto df = vsubq_f32(v, cb);
                        acc = vmlaq_f32(acc, df, df);
                    }
                    float dist = vaddvq_f32(acc);
                    for (; d < dim_; ++d) {
                        float diff = vec[d] - cent[d];
                        dist += diff*diff;
                    }
                    if (dist < best) { best = dist; bid = c; }
                }
                assign[i] = bid;
            }
            // gather
            for (size_t i = 0; i < n; ++i) {
                clusters_[assign[i]].push_back((uint32_t)i);
            }
            // update centroids
            #pragma omp parallel for schedule(static)
            for (size_t c = 0; c < N_CLUSTERS; ++c) {
                auto &cls = clusters_[c];
                if (cls.empty()) continue;
                std::vector<float> sum(dim_, 0.0f);
                for (auto idx : cls) {
                    const float* v = data + idx*dim_;
                    size_t d = 0;
                    for (; d+4<=dim_; d+=4) {
                        auto vv = vld1q_f32(v + d);
                        auto ss = vld1q_f32(&sum[d]);
                        ss = vaddq_f32(ss, vv);
                        vst1q_f32(&sum[d], ss);
                    }
                    for (; d<dim_; ++d) sum[d] += v[d];
                }
                float inv = 1.0f / cls.size();
                float* cent = &centroids_[c*dim_];
                size_t d = 0;
                for (; d+4<=dim_; d+=4) {
                    auto ss = vld1q_f32(&sum[d]);
                    ss = vmulq_n_f32(ss, inv);
                    vst1q_f32(cent + d, ss);
                }
                for (; d<dim_; ++d) cent[d] = sum[d] * inv;
            }
        }
    }

    // 找距离 query 最近的 n_probe 个 cluster 的 id
    std::vector<size_t> search_clusters(const float* query,
                                        size_t n_probe) const {
        std::priority_queue<std::pair<float,size_t>> pq;
        for (size_t c = 0; c < N_CLUSTERS; ++c) {
            const float* cent = &centroids_[c*dim_];
            float32x4_t acc = vdupq_n_f32(0.0f);
            size_t d=0;
            for (; d+4<=dim_; d+=4) {
                auto qv = vld1q_f32(query + d);
                auto cb = vld1q_f32(cent + d);
                auto df = vsubq_f32(qv, cb);
                acc = vmlaq_f32(acc, df, df);
            }
            float dist = vaddvq_f32(acc);
            for (; d<dim_; ++d) {
                float diff = query[d] - cent[d];
                dist += diff*diff;
            }
            pq.emplace(dist, c);
            if (pq.size() > n_probe) pq.pop();
        }
        std::vector<size_t> res;
        res.reserve(pq.size());
        while (!pq.empty()) {
            res.push_back(pq.top().second);
            pq.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    const std::vector<uint32_t>& get_cluster(size_t c) const {
        return clusters_[c];
    }
    const float* get_centroid(size_t c) const {
        return centroids_.data() + c*dim_;
    }

    void assign_clusters(const std::vector<uint32_t>& assign,
                         size_t n) {
        clusters_.assign(N_CLUSTERS, {});
        for (size_t i = 0; i < n; ++i) {
            clusters_[assign[i]].push_back((uint32_t)i);
        }
    }
};

// ==== PQIndex w/ IVF + per‐cluster HNSW + Re‐rank ====
class PQIndex {
private:
    // PQ‐codes & codebooks
    std::vector<uint8_t> codes_;                   // [n * SUBSPACE_NUM]
    float*               codebook_[SUBSPACE_NUM];   // [256 × SUBSPACE_DIM]
    size_t               dim_, base_n_{0};
    float*               base_data_ = nullptr;      // 用于 Re‐rank

    // IVF coarse
    IVFIndex             ivf_index_;
    std::vector<uint32_t> assignments_;            // 每个向量的 coarse‐cluster

    // per‐cluster HNSW
    std::vector<hnswlib::HierarchicalNSW<float>*>  cluster_hnsw_;
    std::vector<hnswlib::SpaceInterface<float>*>   cluster_space_;

    // PQSpace for HNSW
    struct PQSpace : public hnswlib::SpaceInterface<float> {
        PQIndex* parent_;
        PQSpace(PQIndex* p) : parent_(p) {}
        size_t get_data_size() override {
            return SUBSPACE_NUM * sizeof(uint8_t);
        }
        hnswlib::DISTFUNC<float> get_dist_func() override {
            return &PQSpace::DistFunc;
        }
        void* get_dist_func_param() override {
            return parent_;
        }
        static float DistFunc(const void* qa,
                              const void* ba,
                              const void* param)
        {
            const uint8_t* code_b = (const uint8_t*)ba;
            PQIndex* P = reinterpret_cast<PQIndex*>(
                            const_cast<void*>(param));
            float dist = 0.0f;
            if (tls_use_asymmetric) {
                const float* lut = tls_lut_ptr;
                for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
                    dist += lut[s*CENTROID_NUM + code_b[s]];
                }
            } else {
                const uint8_t* code_q = (const uint8_t*)qa;
                for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
                    const float* c1 = P->codebook_[s]
                                     + code_q[s]*SUBSPACE_DIM;
                    const float* c2 = P->codebook_[s]
                                     + code_b[s]*SUBSPACE_DIM;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (size_t d = 0; d < SUBSPACE_DIM; d+=4) {
                        auto v1 = vld1q_f32(c1 + d);
                        auto v2 = vld1q_f32(c2 + d);
                        auto df = vsubq_f32(v1, v2);
                        acc = vmlaq_f32(acc, df, df);
                    }
                    dist += vaddvq_f32(acc);
                }
            }
            return dist;
        }
    };

    // ---------- 原 PQ 子空间 KMeans Init ----------
    void kmeans_init(const float* data, size_t n,
                     float* centroids,
                     std::mt19937& rng)
    {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        // 第一个质心
        std::memcpy(centroids,
                    data + idx[0]*SUBSPACE_DIM,
                    SUBSPACE_DIM * sizeof(float));
        std::vector<float> min_d(n, FLT_MAX);

        for (size_t c = 1; c < CENTROID_NUM; ++c) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float* v = data + i*SUBSPACE_DIM;
                float best = min_d[i];
                for (size_t pc = 0; pc < c; ++pc) {
                    const float* cent = centroids + pc*SUBSPACE_DIM;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (size_t d = 0; d < SUBSPACE_DIM; d += 4) {
                        auto vv = vld1q_f32(v + d);
                        auto cc = vld1q_f32(cent + d);
                        auto df = vsubq_f32(vv, cc);
                        acc = vmlaq_f32(acc, df, df);
                    }
                    float dist = vaddvq_f32(acc);
                    if (dist < best) best = dist;
                }
                min_d[i] = best;
            }
            // 累积概率抽样下一个质心
            std::vector<float> cum(n);
            std::partial_sum(min_d.begin(), min_d.end(), cum.begin());
            std::uniform_real_distribution<float> dis(0.0f, cum.back());
            float r = dis(rng);
            size_t sel = std::lower_bound(cum.begin(), cum.end(), r)
                         - cum.begin();
            if (sel >= n) sel = n-1;
            std::memcpy(centroids + c*SUBSPACE_DIM,
                        data     + sel*SUBSPACE_DIM,
                        SUBSPACE_DIM*sizeof(float));
        }
    }

public:
    PQIndex(size_t dim = DIM)
      : dim_(dim), ivf_index_(dim)
    {
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 16,
                   CENTROID_NUM*SUBSPACE_DIM*sizeof(float))) {
                throw std::bad_alloc();
            }
            codebook_[s] = reinterpret_cast<float*>(ptr);
        }
    }
    ~PQIndex() {
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) free(codebook_[s]);
        for (auto p : cluster_hnsw_)  delete p;
        for (auto p : cluster_space_) delete p;
    }

    // ============ 1) 训练 IVF + PQ ============
    void train(float* data, size_t n) {
        ivf_index_.build(data, n);
        assignments_.resize(n);
        std::vector<float> residuals(n * dim_);

        // compute residuals
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t cid = ivf_index_.search_clusters(
                              data + i*dim_,1)[0];
            assignments_[i] = (uint32_t)cid;
            const float* cent = ivf_index_.get_centroid(cid);
            float*       res  = residuals.data() + i*dim_;
            for (size_t d = 0; d < dim_; ++d)
                res[d] = data[i*dim_ + d] - cent[d];
        }

        // PQ 子空间 KMeans
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            std::vector<float> sub(n*SUBSPACE_DIM);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::memcpy(sub.data()+i*SUBSPACE_DIM,
                            residuals.data()+i*dim_ + s*SUBSPACE_DIM,
                            SUBSPACE_DIM*sizeof(float));
            }
            std::mt19937 rng((unsigned)s);
            kmeans_init(sub.data(), n, codebook_[s], rng);

            // kmeans iterate
            std::vector<size_t> assign(n);
            std::vector<float>  sum(CENTROID_NUM*SUBSPACE_DIM);
            std::vector<size_t> cnt(CENTROID_NUM);

            for (size_t it = 0; it < KMEANS_ITER; ++it) {
                std::fill(sum.begin(), sum.end(), 0.0f);
                std::fill(cnt.begin(), cnt.end(), 0);
                #pragma omp parallel
                {
                    std::vector<float>  loc_sum(CENTROID_NUM*SUBSPACE_DIM);
                    std::vector<size_t> loc_cnt(CENTROID_NUM);

                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < n; ++i) {
                        const float* v = sub.data() + i*SUBSPACE_DIM;
                        float best = FLT_MAX; size_t bid = 0;
                        for (size_t c = 0; c < CENTROID_NUM; ++c) {
                            const float* cent =
                              codebook_[s] + c*SUBSPACE_DIM;
                            float32x4_t acc = vdupq_n_f32(0.0f);
                            for (size_t d = 0; d < SUBSPACE_DIM; d+=4) {
                                auto vv = vld1q_f32(v + d);
                                auto cc = vld1q_f32(cent + d);
                                auto df = vsubq_f32(vv, cc);
                                acc = vmlaq_f32(acc, df, df);
                            }
                            float dist = vaddvq_f32(acc);
                            if (dist < best) { best = dist; bid = c; }
                        }
                        assign[i] = bid;
                    }
                    #pragma omp for nowait
                    for (size_t i = 0; i < n; ++i) {
                        size_t c = assign[i];
                        const float* v = sub.data() + i*SUBSPACE_DIM;
                        for (size_t d = 0; d < SUBSPACE_DIM; ++d)
                            loc_sum[c*SUBSPACE_DIM + d] += v[d];
                        loc_cnt[c]++;
                    }
                    #pragma omp critical
                    {
                        for (size_t c = 0; c < CENTROID_NUM; ++c) {
                            cnt[c] += loc_cnt[c];
                            for (size_t d = 0; d < SUBSPACE_DIM; ++d)
                                sum[c*SUBSPACE_DIM + d] +=
                                    loc_sum[c*SUBSPACE_DIM + d];
                        }
                    }
                }
                std::uniform_int_distribution<size_t> uid(0, n-1);
                for (size_t c = 0; c < CENTROID_NUM; ++c) {
                    float* cent = codebook_[s] + c*SUBSPACE_DIM;
                    if (cnt[c] == 0) {
                        size_t r = uid(rng);
                        std::memcpy(cent,
                                    sub.data() + r*SUBSPACE_DIM,
                                    SUBSPACE_DIM*sizeof(float));
                    } else {
                        for (size_t d = 0; d < SUBSPACE_DIM; ++d) {
                            cent[d] = sum[c*SUBSPACE_DIM + d] / cnt[c];
                        }
                    }
                }
            }
        }
    }

    // ============ 2) 编码 + per‐cluster HNSW 构建 ============
    void encode(float* data, size_t n) {
        base_n_      = n;
        base_data_   = data;             // for Re‐rank
        codes_.resize(n*SUBSPACE_NUM);
        assignments_.resize(n);

        // 2.1 计算 assignments_ & PQ codes
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            // coarse assign
            size_t cid = ivf_index_.search_clusters(
                              data + i*dim_,1)[0];
            assignments_[i] = (uint32_t)cid;
            // residual
            float local_res[DIM];
            const float* cent = ivf_index_.get_centroid(cid);
            for (size_t d = 0; d < dim_; ++d)
                local_res[d] = data[i*dim_+d] - cent[d];
            // PQ encode
            for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
                const float* subv = local_res + s*SUBSPACE_DIM;
                float best = FLT_MAX; uint8_t bestc = 0;
                for (size_t c = 0; c < CENTROID_NUM; ++c) {
                    const float* cent_s =
                      codebook_[s] + c*SUBSPACE_DIM;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (size_t d = 0; d < SUBSPACE_DIM; d+=4) {
                        auto v1 = vld1q_f32(subv + d);
                        auto v2 = vld1q_f32(cent_s + d);
                        auto df = vsubq_f32(v1, v2);
                        acc = vmlaq_f32(acc, df, df);
                    }
                    float dist = vaddvq_f32(acc);
                    if (dist < best) {
                        best = dist; bestc = (uint8_t)c;
                    }
                }
                codes_[i*SUBSPACE_NUM + s] = bestc;
            }
        }

        // 2.2 更新 IVFIndex.clusters_
        ivf_index_.assign_clusters(assignments_, n);

        // 2.3 构建 per‐cluster HNSW
        std::vector<std::vector<uint32_t>> tmp(N_CLUSTERS);
        for (size_t i = 0; i < n; ++i)
            tmp[assignments_[i]].push_back((uint32_t)i);

        cluster_space_.assign(N_CLUSTERS, nullptr);
        cluster_hnsw_.assign(N_CLUSTERS, nullptr);

        for (size_t c = 0; c < N_CLUSTERS; ++c) {
            auto &bucket = tmp[c];
            if (bucket.empty()) continue;
            auto* space = new PQSpace(this);
            auto* index = new hnswlib::HierarchicalNSW<float>(
                              space,
                              bucket.size(),
                              /*M=*/HNSW_M,
                              /*ef_c=*/HNSW_EFC);
            index->ef_ = HNSW_EFSEARCH;
            for (auto idx : bucket) {
                index->addPoint(codes_.data() + idx*SUBSPACE_NUM, idx);
            }
            cluster_space_[c] = space;
            cluster_hnsw_[c]  = index;
        }
    }

    const IVFIndex& get_ivf()    const { return ivf_index_; }
    bool has_hnsw()              const {
        for (auto *h : cluster_hnsw_) if (h) return true;
        return false;
    }
    const uint8_t* get_codes()   const { return codes_.data(); }
    const float*   get_base_data() const { return base_data_; }

    // 3) HNSW 搜索 w/ IVF+Asymmetric‐PQ
    std::priority_queue<std::pair<float,uint32_t>>
    search_hnsw(const float* query, size_t k) const {
        auto probe = ivf_index_.search_clusters(query, N_PROBE);
        int nthreads = omp_get_max_threads();
        std::vector<std::priority_queue<std::pair<float,uint32_t>>>
            local_q(nthreads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& myq = local_q[tid];
            float local_res[DIM];
            float my_lut[SUBSPACE_NUM * CENTROID_NUM];
            uint8_t qcode[SUBSPACE_NUM];

            #pragma omp for schedule(dynamic)
            for (size_t pi = 0; pi < probe.size(); ++pi) {
                size_t c = probe[pi];
                auto* sub_index = cluster_hnsw_[c];
                if (!sub_index) continue;

                // 3.1 residual
                const float* cent = ivf_index_.get_centroid(c);
                for (size_t d = 0; d < dim_; ++d)
                    local_res[d] = query[d] - cent[d];

                // 3.2 LUT
                precompute_LUT(local_res, my_lut);

                // 3.3 build qcode (symmetric)
                for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
                    const float* subv = local_res + s*SUBSPACE_DIM;
                    float best = FLT_MAX; uint8_t bestc=0;
                    for (size_t cc = 0; cc < CENTROID_NUM; ++cc) {
                        const float* cent_s =
                          codebook_[s] + cc*SUBSPACE_DIM;
                        float32x4_t acc = vdupq_n_f32(0.0f);
                        for (size_t d = 0; d < SUBSPACE_DIM; d+=4) {
                            auto v1 = vld1q_f32(subv + d);
                            auto v2 = vld1q_f32(cent_s + d);
                            auto df = vsubq_f32(v1, v2);
                            acc = vmlaq_f32(acc, df, df);
                        }
                        float dist = vaddvq_f32(acc);
                        if (dist < best) { best = dist; bestc=(uint8_t)cc; }
                    }
                    qcode[s] = bestc;
                }

                // 3.4 set TLS for asymmetric dist
                tls_lut_ptr        = my_lut;
                tls_use_asymmetric = true;

                // 3.5 HNSW KNN search
                auto heap = sub_index->searchKnn(qcode, k);
                while (!heap.empty()) {
                    myq.emplace(heap.top().first,
                                (uint32_t)heap.top().second);
                    if (myq.size() > k) myq.pop();
                    heap.pop();
                }
            }
        }

        // 合并线程局部 heap
        std::priority_queue<std::pair<float,uint32_t>> final_q;
        for (auto &tq : local_q) {
            while (!tq.empty()) {
                final_q.push(tq.top());
                if (final_q.size() > k) final_q.pop();
                tq.pop();
            }
        }
        return final_q;
    }

    // 4) precompute LUT for asymmetric‐PQ
    void precompute_LUT(const float* query, float* out_lut) const {
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            for (size_t c = 0; c < CENTROID_NUM; ++c) {
                const float* cent = codebook_[s] + c*SUBSPACE_DIM;
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (size_t d = 0; d < SUBSPACE_DIM; d+=4) {
                    auto qv = vld1q_f32(query + s*SUBSPACE_DIM + d);
                    auto cc = vld1q_f32(cent + d);
                    auto df = vsubq_f32(qv, cc);
                    acc = vmlaq_f32(acc, df, df);
                }
                out_lut[s*CENTROID_NUM + c] = vaddvq_f32(acc);
            }
        }
    }
};

// ==== 全局 pq_search fallback: IVF + PQ 扫描 ====
static std::priority_queue<std::pair<float,uint32_t>>
pq_search(const PQIndex& index,
          const float* query,
          size_t /*base_number*/,
          size_t k)
{
    if (index.has_hnsw()) {
        return index.search_hnsw(query, k);
    }
    // IVF probe
    auto probe = index.get_ivf().search_clusters(query, N_PROBE);

    thread_local float lut_full[SUBSPACE_NUM * CENTROID_NUM];
    size_t nth = omp_get_max_threads();
    std::vector<std::priority_queue<std::pair<float,uint32_t>>> local_q(nth);

    #pragma omp parallel for schedule(dynamic)
    for (size_t pi = 0; pi < probe.size(); ++pi) {
        size_t c = probe[pi];
        // residual + LUT
        float local_res[DIM];
        const float* cent = index.get_ivf().get_centroid(c);
        for (size_t d = 0; d < DIM; ++d)
            local_res[d] = query[d] - cent[d];
        index.precompute_LUT(local_res, lut_full);

        // scan bucket
        const auto& bucket = index.get_ivf().get_cluster(c);
        auto& q = local_q[omp_get_thread_num()];
        for (auto idx : bucket) {
            const uint8_t* code = index.get_codes() + idx*SUBSPACE_NUM;
            float dist = 0.0f;
            for (size_t s = 0; s < SUBSPACE_NUM; ++s)
                dist += lut_full[s*CENTROID_NUM + code[s]];
            if (q.size() < k || dist < q.top().first) {
                q.emplace(dist, idx);
                if (q.size() > k) q.pop();
            }
        }
    }
    // merge
    std::priority_queue<std::pair<float,uint32_t>> final_q;
    for (auto& q : local_q) {
        while (!q.empty()) {
            final_q.push(q.top());
            if (final_q.size() > k) final_q.pop();
            q.pop();
        }
    }
    return final_q;
}

// ==== exact L2 距离，用于 Re‐rank ====
static float exact_l2(const float* a,
                      const float* b,
                      size_t dim)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d+4<=dim; d+=4) {
        auto va = vld1q_f32(a + d);
        auto vb = vld1q_f32(b + d);
        auto df = vsubq_f32(va, vb);
        acc = vmlaq_f32(acc, df, df);
    }
    float sum = vaddvq_f32(acc);
    for (; d<dim; ++d) {
        float df = a[d] - b[d];
        sum += df*df;
    }
    return sum;
}

// ==== batch_search w/ Re‐rank （C++11/C++14 版） ====
static void batch_search(const PQIndex& index,
                         const float*   queries,
                         size_t         Q,
                         size_t         K,
                         std::vector<std::vector<std::pair<float,uint32_t> > >& results)
{
    // 动态取 dim
    const float* c0 = index.get_ivf().get_centroid(0);
    const float* c1 = index.get_ivf().get_centroid(1);
    size_t dim = size_t(c1 - c0);

    results.resize(Q);
    #pragma omp parallel for schedule(dynamic)
    for (size_t qi = 0; qi < Q; ++qi) {
        const float* qptr = queries + qi * dim;

        // 第一步 PQ 搜 top‐L
        size_t L = K * RERANK_FACTOR;
        std::priority_queue<std::pair<float,uint32_t> > heap =
            pq_search(index, qptr, /*base=*/0, L);

        // 第二步 Re‐rank：计算原始精确距
        std::vector<std::pair<float,uint32_t> > rerank;
        rerank.reserve(L);
        const float* base = index.get_base_data();
        // 原来写成 structured binding，改成下面形式：
        while (!heap.empty()) {
            // pop one candidate
            std::pair<float,uint32_t> pr = heap.top();
            heap.pop();
            float est = pr.first;
            uint32_t idx = pr.second;
            // 计算真实距离
            float reald = exact_l2(qptr, base + idx*dim, dim);
            rerank.push_back(std::make_pair(reald, idx));
        }
        // 排序并截断到 K。原来用 auto 泛型 lambda，这里改成显式类型：
        std::sort(rerank.begin(), rerank.end(),
                  // 显式写出类型 pair<float,uint32_t>
                  (bool(*)(const std::pair<float,uint32_t>&,
                          const std::pair<float,uint32_t>&))
                  ([](const std::pair<float,uint32_t>& a,
                      const std::pair<float,uint32_t>& b)->bool{
                       return a.first < b.first;
                   })
        );
        if (rerank.size() > K) {
            rerank.resize(K);
        }
        results[qi].swap(rerank);
    }
}
