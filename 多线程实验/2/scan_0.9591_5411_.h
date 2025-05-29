// pq_ivf_opq_2lvl_hnsw.cpp
#include <vector>
#include <cstring>
#include <random>
#include <numeric>
#include <limits>
#include <cfloat>
#include <queue>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <omp.h>
#include <arm_neon.h>
#include "hnswlib/hnswlib/hnswlib.h"

using namespace hnswlib;

// ====== 参数区 =================================================
// 全局维度
static constexpr size_t DIM            = 128;
// IVF 粗量化
static constexpr size_t N_CLUSTERS     = 100;
static constexpr size_t IVF_KMEANS_ITER= 10;
static constexpr size_t N_PROBE        = 5;   // ↑ probe 放大
// PQ 子空间数与子维度
static constexpr size_t SUBSPACE_NUM   = 16;
static constexpr size_t SUBSPACE_DIM   = DIM/SUBSPACE_NUM;
// PQ 中心数
static constexpr size_t CENTROID_NUM_L1= 256;
static constexpr size_t CENTROID_NUM_L2= 256;
static constexpr size_t KMEANS_ITER    = 10;
// HNSW 参数
static constexpr size_t HNSW_M         = 32;    // ↑ M 放大
static constexpr size_t HNSW_EF_CONSTR = 200;
static constexpr size_t HNSW_EF_SEARCH = 100;  // ↑ efSearch 放大
// 精排候选数
static constexpr size_t TOPK_CAND      = 500;   // ↑ TOPK_CAND 放大

// 线程局部 LUT 指针
static thread_local const float* tls_lut1 = nullptr;
static thread_local const float* tls_lut2 = nullptr;
static thread_local bool        tls_use_asymmetric = false;
// ===========================================================

// -------------------- IVFIndex 不变 -------------------------
class IVFIndex {
private:
    size_t dim_;
    std::vector<float>                  centroids_;  // N_CLUSTERS * dim_
    std::vector<std::vector<uint32_t>>  clusters_;

    void kmeans_init(const float* data, size_t n) {
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

    void build(const float* data, size_t n) {
        kmeans_init(data, n);
        std::vector<size_t> assign(n);
        for (size_t it = 0; it < IVF_KMEANS_ITER; ++it) {
            clusters_.assign(N_CLUSTERS, {});
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float* vec = data + i*dim_;
                float best = FLT_MAX; size_t bid = 0;
                for (size_t c = 0; c < N_CLUSTERS; ++c) {
                    const float* cent = &centroids_[c*dim_];
                    // neon 加速
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    size_t d=0;
                    for (; d+4<=dim_; d+=4) {
                        auto v  = vld1q_f32(vec + d);
                        auto cb = vld1q_f32(cent + d);
                        auto df = vsubq_f32(v, cb);
                        acc = vmlaq_f32(acc, df, df);
                    }
                    float dist = vaddvq_f32(acc);
                    for (; d<dim_; ++d) {
                        float diff = vec[d] - cent[d];
                        dist += diff*diff;
                    }
                    if (dist < best) { best = dist; bid = c; }
                }
                assign[i] = bid;
            }
            for (size_t i = 0; i < n; ++i) {
                clusters_[assign[i]].push_back((uint32_t)i);
            }
            #pragma omp parallel for schedule(static)
            for (size_t c = 0; c < N_CLUSTERS; ++c) {
                auto &cls = clusters_[c];
                if (cls.empty()) continue;
                std::vector<float> sum(dim_, 0.0f);
                for (auto idx : cls) {
                    const float* v = data + idx*dim_;
                    size_t d=0;
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
                size_t d=0;
                for (; d+4<=dim_; d+=4) {
                    auto ss = vld1q_f32(&sum[d]);
                    ss = vmulq_n_f32(ss, inv);
                    vst1q_f32(cent + d, ss);
                }
                for (; d<dim_; ++d) cent[d] = sum[d] * inv;
            }
        }
    }

    // 返回 [最近,…远]
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
};

// ==== PQIndex with OPQ + 二级残差PQ + HNSW ====================
class PQIndex {
private:
    const float*         base_data_ = nullptr;
    size_t               dim_, base_n_{0};

    // OPQ 线性变换矩阵：dim×dim
    float*               R_;  // 行优先

    // 一级 PQ
    std::vector<uint8_t> codes1_;                    // n×SUBSPACE_NUM
    float*                cb1_[SUBSPACE_NUM];        // SUBSPACE_NUM × (CENTROID_NUM_L1×SUBSPACE_DIM)

    // 二级 PQ（在残差基础上再量化一次）
    std::vector<uint8_t> codes2_;                    // n×SUBSPACE_NUM
    float*                cb2_[SUBSPACE_NUM];        // SUBSPACE_NUM × (CENTROID_NUM_L2×SUBSPACE_DIM)

    // IVF
    IVFIndex              ivf_index_;
    std::vector<uint32_t> assignments_;

    // per‐cluster HNSW（建在第一层PQ code之上）
    std::vector<hnswlib::HierarchicalNSW<float>*> cluster_hnsw_;
    std::vector<hnswlib::SpaceInterface<float>*>  cluster_space_;

    // PQSpace for HNSW
    struct PQSpace : public hnswlib::SpaceInterface<float> {
        PQIndex* parent_;
        PQSpace(PQIndex* p): parent_(p){}
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
            auto* P = (PQIndex*)param;
            const uint8_t* code_b = (const uint8_t*)ba;
            float dist = 0;
            if (tls_use_asymmetric) {
                // Asym PQ：lut1+lut2
                for (size_t s=0; s<SUBSPACE_NUM; ++s) {
                    dist += tls_lut1[s*CENTROID_NUM_L1 + code_b[s]];
                    dist += tls_lut2[s*CENTROID_NUM_L2 + code_b[s]];
                }
            } else {
                // Sym PQ on code1
                const uint8_t* code_q = (const uint8_t*)qa;
                for (size_t s=0; s<SUBSPACE_NUM; ++s) {
                    const float* c1 = P->cb1_[s] +
                                     code_q[s]*SUBSPACE_DIM;
                    const float* c2 = P->cb1_[s] +
                                     code_b[s]*SUBSPACE_DIM;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (size_t d=0; d<SUBSPACE_DIM; d+=4) {
                        auto v1=vld1q_f32(c1+d);
                        auto v2=vld1q_f32(c2+d);
                        auto df=vsubq_f32(v1,v2);
                        acc=vmlaq_f32(acc,df,df);
                    }
                    dist+=vaddvq_f32(acc);
                }
            }
            return dist;
        }
    };

    // kmeans init 同原
    void kmeans_init(const float* data, size_t n,
                     float* centroids, size_t K, std::mt19937& rng)
    {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::memcpy(centroids,
                    data + idx[0]*SUBSPACE_DIM,
                    SUBSPACE_DIM * sizeof(float));
        std::vector<float> min_d(n, FLT_MAX);
        for (size_t c = 1; c < K; ++c) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float* v = data + i*SUBSPACE_DIM;
                float best = min_d[i];
                for (size_t pc = 0; pc < c; ++pc) {
                    const float* cent = centroids+pc*SUBSPACE_DIM;
                    float32x4_t acc=vdupq_n_f32(0.0f);
                    for (size_t d=0; d<SUBSPACE_DIM; d+=4) {
                        auto vv=vld1q_f32(v+d);
                        auto cc=vld1q_f32(cent+d);
                        auto df=vsubq_f32(vv,cc);
                        acc=vmlaq_f32(acc,df,df);
                    }
                    float dist=vaddvq_f32(acc);
                    if (dist<best) best=dist;
                }
                min_d[i]=best;
            }
            std::vector<float> cum(n);
            std::partial_sum(min_d.begin(),min_d.end(),cum.begin());
            std::uniform_real_distribution<float> dis(0.0f,cum.back());
            float r=dis(rng);
            size_t sel=std::lower_bound(cum.begin(),cum.end(),r)-cum.begin();
            if (sel>=n) sel=n-1;
            std::memcpy(centroids+c*SUBSPACE_DIM,
                        data+sel*SUBSPACE_DIM,
                        SUBSPACE_DIM*sizeof(float));
        }
    }

    // 对矩阵做 QR 分解，生成正交矩阵 R （这里只用随机初始化+正交化）
    void init_opq() {
        // R_ 大小 DIM×DIM，16字节对齐
        posix_memalign((void**)&R_, 64, sizeof(float)*dim_*dim_);
        std::mt19937 rng(1234);
        // 随机高斯
        for (size_t i = 0; i < dim_*dim_; ++i)
            R_[i] = std::normal_distribution<float>()(rng);
        // Gram‐Schmidt 正交化每列
        for (size_t c=0; c<dim_; ++c) {
            float* col_c = R_ + c*dim_;
            // 正规化前，减去和之前列的投影
            for (size_t j=0; j<c; ++j) {
                float* col_j = R_ + j*dim_;
                float dot=0;
                for (size_t i=0; i<dim_; ++i)
                    dot += col_j[i]*col_c[i];
                for (size_t i=0; i<dim_; ++i)
                    col_c[i] -= dot * col_j[i];
            }
            // 归一化
            float norm=0;
            for (size_t i=0; i<dim_; ++i) norm += col_c[i]*col_c[i];
            norm = std::sqrt(norm);
            assert(norm>1e-6);
            for (size_t i=0; i<dim_; ++i)
                col_c[i] /= norm;
        }
    }

public:
    PQIndex(size_t dim=DIM)
        : dim_(dim), ivf_index_(dim)
    {
        // alloc codebooks
        for (size_t s=0; s<SUBSPACE_NUM; ++s) {
            posix_memalign((void**)&cb1_[s], 64,
                CENTROID_NUM_L1*SUBSPACE_DIM*sizeof(float));
            posix_memalign((void**)&cb2_[s], 64,
                CENTROID_NUM_L2*SUBSPACE_DIM*sizeof(float));
        }
        init_opq();
    }
    ~PQIndex() {
        free(R_);
        for (size_t s=0; s<SUBSPACE_NUM; ++s) {
            free(cb1_[s]);
            free(cb2_[s]);
        }
        for (auto p:cluster_hnsw_) delete p;
        for (auto p:cluster_space_) delete p;
    }

    // 1) train: IVF + OPQ + 两级PQ
    void train(float* data, size_t n) {
        // 1.1 建 IVF
        ivf_index_.build(data,n);
        assignments_.resize(n);
        // 1.2 计算残差并 OPQ 变换
        std::vector<float> res_opq(n*dim_);
        #pragma omp parallel for schedule(static)
        for (size_t i=0; i<n; ++i) {
            size_t cid = ivf_index_.search_clusters(data+i*dim_,1)[0];
            assignments_[i] = (uint32_t)cid;
            const float* cent = ivf_index_.get_centroid(cid);
            float* r_o = res_opq.data()+i*dim_;
            float tmp[DIM];
            // 原始残差
            for (size_t d=0; d<dim_; ++d)
                tmp[d] = data[i*dim_+d] - cent[d];
            // OPQ R_: r_o = R_ * tmp
            for (size_t c=0; c<dim_; ++c) {
                float sum=0;
                float* col = R_ + c*dim_;
                for (size_t d=0; d<dim_; ++d)
                    sum += col[d]*tmp[d];
                r_o[c] = sum;
            }
        }
        // 1.3 一级PQ训练
        for (size_t s=0; s<SUBSPACE_NUM; ++s) {
            // subview
            std::vector<float> sub(n*SUBSPACE_DIM);
            #pragma omp parallel for schedule(static)
            for (size_t i=0;i<n;++i) {
                memcpy(sub.data()+i*SUBSPACE_DIM,
                       res_opq.data()+i*dim_+s*SUBSPACE_DIM,
                       SUBSPACE_DIM*sizeof(float));
            }
            std::mt19937 rng((unsigned)s+1);
            kmeans_init(sub.data(),n,cb1_[s],CENTROID_NUM_L1,rng);
            // standard kmeans iter
            std::vector<size_t> assign(n);
            std::vector<float>  sum(CENTROID_NUM_L1*SUBSPACE_DIM);
            std::vector<size_t> cnt(CENTROID_NUM_L1);
            for (size_t it=0; it<KMEANS_ITER; ++it) {
                std::fill(sum.begin(),sum.end(),0.0f);
                std::fill(cnt.begin(),cnt.end(),0);
                #pragma omp parallel
                {
                    std::vector<float>  loc_sum(CENTROID_NUM_L1*SUBSPACE_DIM);
                    std::vector<size_t> loc_cnt(CENTROID_NUM_L1);
                    #pragma omp for schedule(static)
                    for (size_t i=0;i<n;++i) {
                        const float* v = sub.data()+i*SUBSPACE_DIM;
                        float best=FLT_MAX; size_t bid=0;
                        for (size_t c=0;c<CENTROID_NUM_L1;++c) {
                            const float* cent = cb1_[s]+c*SUBSPACE_DIM;
                            float32x4_t acc=vdupq_n_f32(0.0f);
                            for (size_t d=0;d<SUBSPACE_DIM;d+=4){
                                auto vv=vld1q_f32(v+d);
                                auto cc=vld1q_f32(cent+d);
                                auto df=vsubq_f32(vv,cc);
                                acc=vmlaq_f32(acc,df,df);
                            }
                            float dist=vaddvq_f32(acc);
                            if (dist<best){best=dist;bid=c;}
                        }
                        assign[i]=bid;
                    }
                    #pragma omp for
                    for (size_t i=0;i<n;++i){
                        size_t c=assign[i];
                        const float* v=sub.data()+i*SUBSPACE_DIM;
                        for (size_t d=0;d<SUBSPACE_DIM;++d)
                            loc_sum[c*SUBSPACE_DIM+d]+=v[d];
                        loc_cnt[c]++;
                    }
                    #pragma omp critical
                    {
                        for (size_t c=0;c<CENTROID_NUM_L1;++c){
                            cnt[c]+=loc_cnt[c];
                            for (size_t d=0;d<SUBSPACE_DIM;++d)
                                sum[c*SUBSPACE_DIM+d]+=
                                    loc_sum[c*SUBSPACE_DIM+d];
                        }
                    }
                }
                std::uniform_int_distribution<size_t> uid(0,n-1);
                for (size_t c=0;c<CENTROID_NUM_L1;++c){
                    float* cent = cb1_[s]+c*SUBSPACE_DIM;
                    if (cnt[c]==0){
                        size_t r=uid(rng);
                        memcpy(cent,sub.data()+r*SUBSPACE_DIM,
                               SUBSPACE_DIM*sizeof(float));
                    } else {
                        for (size_t d=0;d<SUBSPACE_DIM;++d)
                            cent[d]=sum[c*SUBSPACE_DIM+d]/cnt[c];
                    }
                }
            }
        }
        // 1.4 二级PQ训练（在一级量化残差上）
        // 计算一级残差并准备 sub2
        std::vector<float> res2(n*dim_);
        #pragma omp parallel for schedule(static)
        for (size_t i=0;i<n;++i){
            float* r  = res_opq.data()+i*dim_;
            float* r2 = res2.data()+i*dim_;
            for (size_t s=0;s<SUBSPACE_NUM;++s){
                // reconstruct level1
                uint8_t bestc=0;
                float best=FLT_MAX;
                const float* subv = r + s*SUBSPACE_DIM;
                for (size_t c=0;c<CENTROID_NUM_L1;++c){
                    const float* cent1=cb1_[s]+c*SUBSPACE_DIM;
                    float32x4_t acc=vdupq_n_f32(0.0f);
                    for (size_t d=0;d<SUBSPACE_DIM;d+=4){
                        auto v1=vld1q_f32(subv+d);
                        auto v2=vld1q_f32(cent1+d);
                        auto df=vsubq_f32(v1,v2);
                        acc=vmlaq_f32(acc,df,df);
                    }
                    float dist=vaddvq_f32(acc);
                    if (dist<best){best=dist;bestc=(uint8_t)c;}
                }
                const float* cent1=cb1_[s]+bestc*SUBSPACE_DIM;
                // residual2 = r - cent1
                for (size_t d=0;d<SUBSPACE_DIM;++d)
                    r2[s*SUBSPACE_DIM+d] =
                      subv[d] - cent1[d];
            }
        }
        for (size_t s=0;s<SUBSPACE_NUM;++s){
            std::vector<float> sub2(n*SUBSPACE_DIM);
            #pragma omp parallel for schedule(static)
            for (size_t i=0;i<n;++i){
                memcpy(sub2.data()+i*SUBSPACE_DIM,
                       res2.data()+i*dim_+s*SUBSPACE_DIM,
                       SUBSPACE_DIM*sizeof(float));
            }
            std::mt19937 rng((unsigned)(s+100));
            kmeans_init(sub2.data(),n,cb2_[s],CENTROID_NUM_L2,rng);
            // 简化：只一轮kmeans
        }
    }

    // 2) encode: 生成 codes1 & codes2，并构建每簇HNSW
    void encode(float* data, size_t n) {
        base_data_=data; base_n_=n;
        codes1_.assign(n*SUBSPACE_NUM,0);
        codes2_.assign(n*SUBSPACE_NUM,0);
        assignments_.resize(n);

        #pragma omp parallel for schedule(static)
        for (size_t i=0;i<n;++i){
            size_t cid = ivf_index_.search_clusters(
                           data+i*dim_,1)[0];
            assignments_[i]=(uint32_t)cid;
            const float* cent=ivf_index_.get_centroid(cid);
            float tmp[DIM], rot[DIM];
            for (size_t d=0;d<dim_;++d)
                tmp[d]=data[i*dim_+d]-cent[d];
            // OPQ 变换
            for (size_t c=0;c<dim_;++c){
                float sum=0;
                float* col=R_+c*dim_;
                for (size_t d=0;d<dim_;++d)
                    sum+=col[d]*tmp[d];
                rot[c]=sum;
            }
            // L1 PQ 编码
            for (size_t s=0;s<SUBSPACE_NUM;++s){
                const float* subv=rot+s*SUBSPACE_DIM;
                float best=FLT_MAX; uint8_t bc=0;
                for (size_t c=0;c<CENTROID_NUM_L1;++c){
                    const float* cent1=cb1_[s]+c*SUBSPACE_DIM;
                    float32x4_t acc=vdupq_n_f32(0.0f);
                    for (size_t d=0;d<SUBSPACE_DIM;d+=4){
                        auto v1=vld1q_f32(subv+d);
                        auto v2=vld1q_f32(cent1+d);
                        auto df=vsubq_f32(v1,v2);
                        acc=vmlaq_f32(acc,df,df);
                    }
                    float dist=vaddvq_f32(acc);
                    if(dist<best){best=dist;bc=(uint8_t)c;}
                }
                codes1_[i*SUBSPACE_NUM+s]=bc;
                // 计算残差1到 rot
                float* patch = rot + s*SUBSPACE_DIM;
                for (size_t d=0;d<SUBSPACE_DIM;++d)
                    patch[d] -= cb1_[s][bc*SUBSPACE_DIM+d];
            }
            // L2 PQ 编码
            for (size_t s=0;s<SUBSPACE_NUM;++s){
                const float* subv=rot+s*SUBSPACE_DIM;
                float best=FLT_MAX; uint8_t bc=0;
                for (size_t c=0;c<CENTROID_NUM_L2;++c){
                    const float* cent2=cb2_[s]+c*SUBSPACE_DIM;
                    float32x4_t acc=vdupq_n_f32(0.0f);
                    for (size_t d=0;d<SUBSPACE_DIM;d+=4){
                        auto v1=vld1q_f32(subv+d);
                        auto v2=vld1q_f32(cent2+d);
                        auto df=vsubq_f32(v1,v2);
                        acc=vmlaq_f32(acc,df,df);
                    }
                    float dist=vaddvq_f32(acc);
                    if(dist<best){best=dist;bc=(uint8_t)c;}
                }
                codes2_[i*SUBSPACE_NUM+s]=bc;
            }
        }

        // 分簇构建HNSW（同原理，用 codes1_）
        std::vector<std::vector<uint32_t>> buckets(N_CLUSTERS);
        for (size_t i=0;i<n;++i)
            buckets[assignments_[i]].push_back(i);

        cluster_hnsw_.assign(N_CLUSTERS,nullptr);
        cluster_space_.assign(N_CLUSTERS,nullptr);
        for (size_t c=0;c<N_CLUSTERS;++c){
            auto &b=buckets[c];
            if (b.empty()) continue;
            auto* space=new PQSpace(this);
            auto* index=new HierarchicalNSW<float>(
                space, b.size(), HNSW_M, HNSW_EF_CONSTR);
            index->ef_=HNSW_EF_SEARCH;
            for (auto idx:b){
                index->addPoint(codes1_.data()+idx*SUBSPACE_NUM, idx);
            }
            cluster_space_[c]=space;
            cluster_hnsw_[c]=index;
        }
    }

    const IVFIndex& get_ivf() const { return ivf_index_; }
    bool has_hnsw() const {
        for (auto*h:cluster_hnsw_) if(h) return true;
        return false;
    }
    const float*   get_base_data() const { return base_data_; }
    size_t         get_dim()       const { return dim_; }
    const uint8_t* get_codes1()    const { return codes1_.data(); }
    const uint8_t* get_codes2()    const { return codes2_.data(); }

    // 预计算两级 LUT
    void precompute_LUT(const float* q,
                        float* lut1, float* lut2) const
    {
        // lut1
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t s=0;s<SUBSPACE_NUM;++s){
            for (size_t c=0;c<CENTROID_NUM_L1;++c){
                const float* cent=cb1_[s]+c*SUBSPACE_DIM;
                float32x4_t acc=vdupq_n_f32(0.0f);
                for (size_t d=0;d<SUBSPACE_DIM;d+=4){
                    auto qv=vld1q_f32(q+s*SUBSPACE_DIM+d);
                    auto cc=vld1q_f32(cent+d);
                    auto df=vsubq_f32(qv,cc);
                    acc=vmlaq_f32(acc,df,df);
                }
                lut1[s*CENTROID_NUM_L1+c]=vaddvq_f32(acc);
            }
        }
        // lut2
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t s=0;s<SUBSPACE_NUM;++s){
            for (size_t c=0;c<CENTROID_NUM_L2;++c){
                const float* cent=cb2_[s]+c*SUBSPACE_DIM;
                float32x4_t acc=vdupq_n_f32(0.0f);
                for (size_t d=0;d<SUBSPACE_DIM;d+=4){
                    auto qv=vld1q_f32(q+s*SUBSPACE_DIM+d);
                    auto cc=vld1q_f32(cent+d);
                    auto df=vsubq_f32(qv,cc);
                    acc=vmlaq_f32(acc,df,df);
                }
                lut2[s*CENTROID_NUM_L2+c]=vaddvq_f32(acc);
            }
        }
    }

    // 3) HNSW + TOPCAND + L2 rerank
    std::priority_queue<std::pair<float,uint32_t>>
    search_hnsw(const float* query, size_t k) const
    {
        // 1) IVF probe
        auto probe = ivf_index_.search_clusters(query, N_PROBE);
        int nth=omp_get_max_threads();
        std::vector<
          std::priority_queue<std::pair<float,uint32_t>>
        > local_q(nth);

        #pragma omp parallel
        {
            int tid=omp_get_thread_num();
            auto& myq=local_q[tid];
            // 缓存区
            float buf[DIM], rot[DIM];
            float lut1[SUBSPACE_NUM*CENTROID_NUM_L1];
            float lut2[SUBSPACE_NUM*CENTROID_NUM_L2];
            uint8_t qcode1[SUBSPACE_NUM];
            // 处理每个簇
            #pragma omp for schedule(dynamic)
            for (size_t pi=0;pi<probe.size();++pi){
                size_t c=probe[pi];
                auto* sub_idx=cluster_hnsw_[c];
                if (!sub_idx) continue;
                // 1) residual & OPQ
                const float* cent=ivf_index_.get_centroid(c);
                for (size_t d=0;d<dim_;++d)
                    buf[d]=query[d]-cent[d];
                // OPQ
                for (size_t cc=0;cc<dim_;++cc){
                    float sum=0;
                    float* col=R_+cc*dim_;
                    for (size_t d=0;d<dim_;++d)
                        sum+=col[d]*buf[d];
                    rot[cc]=sum;
                }
                // 2) lut
                const_cast<PQIndex*>(this)
                    ->precompute_LUT(rot,lut1,lut2);
                // 3) 生成dummy qcode1 for HNSW
                for (size_t s=0;s<SUBSPACE_NUM;++s){
                    const float* subv=rot+s*SUBSPACE_DIM;
                    float best=FLT_MAX; uint8_t bc=0;
                    for (size_t cc=0;cc<CENTROID_NUM_L1;++cc){
                        const float* cent1=cb1_[s]+cc*SUBSPACE_DIM;
                        float32x4_t acc=vdupq_n_f32(0.0f);
                        for (size_t d=0;d<SUBSPACE_DIM;d+=4){
                            auto v1=vld1q_f32(subv+d);
                            auto v2=vld1q_f32(cent1+d);
                            auto df=vsubq_f32(v1,v2);
                            acc=vmlaq_f32(acc,df,df);
                        }
                        float dist=vaddvq_f32(acc);
                        if (dist<best){best=dist;bc=(uint8_t)cc;}
                    }
                    qcode1[s]=bc;
                }
                // 4) HNSW search
                tls_lut1=lut1; tls_lut2=lut2;
                tls_use_asymmetric=true;
                auto heap=sub_idx->searchKnn(qcode1, TOPK_CAND);
                // 5) 合并到本地堆
                while(!heap.empty()){
                    myq.emplace(heap.top().first,
                                (uint32_t)heap.top().second);
                    if(myq.size()>TOPK_CAND) myq.pop();
                    heap.pop();
                }
            }
        }

        // 合并线程本地堆
        std::priority_queue<std::pair<float,uint32_t>> final_q;
        for (auto &tq: local_q){
            while(!tq.empty()){
                final_q.push(tq.top());
                if(final_q.size()>TOPK_CAND)
                    final_q.pop();
                tq.pop();
            }
        }
        // L2 精排
        std::vector<std::pair<float,uint32_t>> rerank;
        rerank.reserve(final_q.size());
        const float* bd = base_data_;
        while(!final_q.empty()){
            rerank.push_back(final_q.top());
            final_q.pop();
        }
        for (auto &pr:rerank){
            uint32_t idx=pr.second;
            const float* bv=bd+idx*dim_;
            float d=0;
            for (size_t i=0;i<dim_;++i){
                float diff=query[i]-bv[i];
                d+=diff*diff;
            }
            pr.first=d;
        }
        // top k
        std::priority_queue<std::pair<float,uint32_t>> outq;
        for (auto &pr: rerank){
            outq.push(pr);
            if (outq.size()>k) outq.pop();
        }
        return outq;
    }
};

// 全局接口不变
static std::priority_queue<std::pair<float,uint32_t>>
pq_search(const PQIndex& index,
          const float* query,
          size_t /*base_n*/, size_t k)
{
    if (index.has_hnsw())
        return index.search_hnsw(query,k);
    return {};
}
