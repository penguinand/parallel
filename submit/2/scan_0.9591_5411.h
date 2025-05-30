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

static constexpr size_t DIM            = 128;

static constexpr size_t N_CLUSTERS     = 100;
static constexpr size_t IVF_KMEANS_ITER= 10;
static constexpr size_t N_PROBE        = 5;

static constexpr size_t SUBSPACE_NUM   = 16;
static constexpr size_t SUBSPACE_DIM   = DIM/SUBSPACE_NUM;

static constexpr size_t CENTROID_NUM_L1= 256;
static constexpr size_t CENTROID_NUM_L2= 256;
static constexpr size_t KMEANS_ITER    = 10;

static constexpr size_t HNSW_M         = 32;
static constexpr size_t HNSW_EF_CONSTR = 200;
static constexpr size_t HNSW_EF_SEARCH = 100;

static constexpr size_t TOPK_CAND      = 500;

static thread_local const float* tls_lut1 = nullptr;
static thread_local const float* tls_lut2 = nullptr;
static thread_local bool        tls_use_asymmetric = false;

class IVFIndex {
private:
    size_t dim_;
    std::vector<float>                  centroids_;
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

class PQIndex {
private:
    const float*         base_data_ = nullptr;
    size_t               dim_, base_n_{0};

    float*               R_;

    std::vector<uint8_t> codes1_;
    float*                cb1_[SUBSPACE_NUM];

    std::vector<uint8_t> codes2_;
    float*                cb2_[SUBSPACE_NUM];

    IVFIndex              ivf_index_;
    std::vector<uint32_t> assignments_;

    std::vector<hnswlib::HierarchicalNSW<float>*> cluster_hnsw_;
    std::vector<hnswlib::SpaceInterface<float>*>  cluster_space_;

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
                for (size_t s=0; s<SUBSPACE_NUM; ++s) {
                    dist += tls_lut1[s*CENTROID_NUM_L1 + code_b[s]];
                    dist += tls_lut2[s*CENTROID_NUM_L2 + code_b[s]];
                }
            } else {
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

    void init_opq() {
        posix_memalign((void**)&R_, 64, sizeof(float)*dim_*dim_);
        std::mt19937 rng(1234);
        for (size_t i = 0; i < dim_*dim_; ++i)
            R_[i] = std::normal_distribution<float>()(rng);
        for (size_t c=0; c<dim_; ++c) {
            float* col_c = R_ + c*dim_;
            for (size_t j=0; j<c; ++j) {
                float* col_j = R_ + j*dim_;
                float dot=0;
                for (size_t i=0; i<dim_; ++i)
                    dot += col_j[i]*col_c[i];
                for (size_t i=0; i<dim_; ++i)
                    col_c[i] -= dot * col_j[i];
            }
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

    void train(float* data, size_t n) {
        ivf_index_.build(data,n);
        assignments_.resize(n);
        std::vector<float> res_opq(n*dim_);
        #pragma omp parallel for schedule(static)
        for (size_t i=0; i<n; ++i) {
            size_t cid = ivf_index_.search_clusters(data+i*dim_,1)[0];
            assignments_[i] = (uint32_t)cid;
            const float* cent = ivf_index_.get_centroid(cid);
            float* r_o = res_opq.data()+i*dim_;
            float tmp[DIM];
            for (size_t d=0; d<dim_; ++d)
                tmp[d] = data[i*dim_+d] - cent[d];
            for (size_t c=0; c<dim_; ++c) {
                float sum=0;
                float* col = R_ + c*dim_;
                for (size_t d=0; d<dim_; ++d)
                    sum += col[d]*tmp[d];
                r_o[c] = sum;
            }
        }
        for (size_t s=0; s<SUBSPACE_NUM; ++s) {
            std::vector<float> sub(n*SUBSPACE_DIM);
            #pragma omp parallel for schedule(static)
            for (size_t i=0;i<n;++i) {
                memcpy(sub.data()+i*SUBSPACE_DIM,
                       res_opq.data()+i*dim_+s*SUBSPACE_DIM,
                       SUBSPACE_DIM*sizeof(float));
            }
            std::mt19937 rng((unsigned)s+1);
            kmeans_init(sub.data(),n,cb1_[s],CENTROID_NUM_L1,rng);
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
        std::vector<float> res2(n*dim_);
        #pragma omp parallel for schedule(static)
        for (size_t i=0;i<n;++i){
            float* r  = res_opq.data()+i*dim_;
            float* r2 = res2.data()+i*dim_;
            for (size_t s=0;s<SUBSPACE_NUM;++s){
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
        }
    }

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
            for (size_t c=0;c<dim_;++c){
                float sum=0;
                float* col=R_+c*dim_;
                for (size_t d=0;d<dim_;++d)
                    sum+=col[d]*tmp[d];
                rot[c]=sum;
            }
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
                float* patch = rot + s*SUBSPACE_DIM;
                for (size_t d=0;d<SUBSPACE_DIM;++d)
                    patch[d] -= cb1_[s][bc*SUBSPACE_DIM+d];
            }
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

    void precompute_LUT(const float* q,
                        float* lut1, float* lut2) const
    {
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

    std::priority_queue<std::pair<float,uint32_t>>
    search_hnsw(const float* query, size_t k) const
    {
        auto probe = ivf_index_.search_clusters(query, N_PROBE);
        int nth=omp_get_max_threads();
        std::vector<
          std::priority_queue<std::pair<float,uint32_t>>
        > local_q(nth);

        #pragma omp parallel
        {
            int tid=omp_get_thread_num();
            auto& myq=local_q[tid];
            float buf[DIM], rot[DIM];
            float lut1[SUBSPACE_NUM*CENTROID_NUM_L1];
            float lut2[SUBSPACE_NUM*CENTROID_NUM_L2];
            uint8_t qcode1[SUBSPACE_NUM];
            #pragma omp for schedule(dynamic)
            for (size_t pi=0;pi<probe.size();++pi){
                size_t c=probe[pi];
                auto* sub_idx=cluster_hnsw_[c];
                if (!sub_idx) continue;
                const float* cent=ivf_index_.get_centroid(c);
                for (size_t d=0;d<dim_;++d)
                    buf[d]=query[d]-cent[d];
                for (size_t cc=0;cc<dim_;++cc){
                    float sum=0;
                    float* col=R_+cc*dim_;
                    for (size_t d=0;d<dim_;++d)
                        sum+=col[d]*buf[d];
                    rot[cc]=sum;
                }
                const_cast<PQIndex*>(this)
                    ->precompute_LUT(rot,lut1,lut2);
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
                tls_lut1=lut1; tls_lut2=lut2;
                tls_use_asymmetric=true;
                auto heap=sub_idx->searchKnn(qcode1, TOPK_CAND);
                while(!heap.empty()){
                    myq.emplace(heap.top().first,
                                (uint32_t)heap.top().second);
                    if(myq.size()>TOPK_CAND) myq.pop();
                    heap.pop();
                }
            }
        }

        std::priority_queue<std::pair<float,uint32_t>> final_q;
        for (auto &tq: local_q){
            while(!tq.empty()){
                final_q.push(tq.top());
                if(final_q.size()>TOPK_CAND)
                    final_q.pop();
                tq.pop();
            }
        }
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
        std::priority_queue<std::pair<float,uint32_t>> outq;
        for (auto &pr: rerank){
            outq.push(pr);
            if (outq.size()>k) outq.pop();
        }
        return outq;
    }
};

static std::priority_queue<std::pair<float,uint32_t>>
pq_search(const PQIndex& index,
          const float* query,
          size_t /*base_n*/, size_t k)
{
    if (index.has_hnsw())
        return index.search_hnsw(query,k);
    return {};
}