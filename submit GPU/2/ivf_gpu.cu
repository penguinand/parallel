#include "ivf_gpu.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <random>
#include <numeric>
#include <queue>
#include <cfloat>
#include <cassert>

#define CUDA_CALL(x)    do{ if((x)!=cudaSuccess){ \
    fprintf(stderr,"CUDA Err %s:%d code=%d\n",__FILE__,__LINE__,x);exit(-1);} }while(0)
#define CUBLAS_CALL(x)  do{ if((x)!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS Err %s:%d code=%d\n",__FILE__,__LINE__,x);exit(-1);} }while(0)

// --- Kernel: 计算 D1[cid, qid] = ||q[qid]-c[cid]||^2
static __global__
void centroid_dist_kernel(
    const float* __restrict__ d_query,   // [dim * m]
    const float* __restrict__ d_cent,    // [dim * C]
    float*       __restrict__ d_D1,      // [C * m]
    int dim, int C, int m)
{
    int cid = blockIdx.x*blockDim.x + threadIdx.x;
    int qid = blockIdx.y*blockDim.y + threadIdx.y;
    if(cid>=C||qid>=m) return;
    const float* cent = d_cent + cid*dim;
    const float* qry  = d_query + qid*dim;
    float s=0;
    #pragma unroll 4
    for(int i=0;i<dim;i++){
        float t = qry[i]-cent[i];
        s += t*t;
    }
    d_D1[cid + qid*C] = s;
}

// --- CPU k-means (init+10轮) ---
static void cpu_kmeans(
    const float* data, size_t n,
    size_t dim, size_t C,
    float* centroids,   // [C*dim]
    uint32_t* csr_ptr,  // [C+1]
    uint32_t* csr_idx)  // [n]
{
    std::mt19937 rng(123);
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    // 随机初始化 C 个中心
    for(size_t c=0;c<C;c++){
        memcpy(centroids + c*dim,
               data + idx[c]*dim,
               sizeof(float)*dim);
    }
    // 临时 clusters
    std::vector<std::vector<uint32_t>> clusters(C);
    for(int it=0; it<10; it++){
        for(auto &cl: clusters) cl.clear();
        // assign
        for(size_t i=0;i<n;i++){
            const float* v = data + i*dim;
            float best=FLT_MAX; int bc=0;
            for(int c=0;c<(int)C;c++){
                const float* ct = centroids + c*dim;
                float ss=0;
                for(size_t d0=0;d0<dim;d0++){
                    float t = v[d0]-ct[d0];
                    ss += t*t;
                }
                if(ss<best){ best=ss; bc=c; }
            }
            clusters[bc].push_back(uint32_t(i));
        }
        // update
        for(int c=0;c<(int)C;c++){
            auto &cl = clusters[c];
            if(cl.empty()) continue;
            for(size_t d0=0; d0<dim; d0++){
                double sum=0;
                for(auto id:cl) sum += data[id*dim+d0];
                centroids[c*dim + d0] = float(sum / cl.size());
            }
        }
    }
    // build CSR
    csr_ptr[0]=0;
    for(size_t c=0;c<C;c++){
        csr_ptr[c+1] = csr_ptr[c] + clusters[c].size();
    }
    // flatten
    for(size_t c=0;c<C;c++){
        auto &cl = clusters[c];
        memcpy(csr_idx + csr_ptr[c],
               cl.data(),
               sizeof(uint32_t)*cl.size());
    }
}

// --- IVFIndex methods ---
IVFIndex::IVFIndex(size_t dim, size_t n_clusters, size_t n_probe)
 : dim_(dim),
   n_clusters_(n_clusters),
   n_probe_(n_probe)
{
    // CPU 缓冲
    h_centroids_ = (float*)malloc(sizeof(float)*n_clusters_*dim_);
    h_csr_ptr_   = (uint32_t*)malloc(sizeof(uint32_t)*(n_clusters_+1));
    // 我们先 build 再 malloc idx
    // 创建 cuBLAS
    CUBLAS_CALL( cublasCreate(&handle_) );
}

IVFIndex::~IVFIndex(){
    free(h_centroids_);
    free(h_csr_ptr_);
    free(h_csr_idx_);
    if(d_base_)      CUDA_CALL(cudaFree(d_base_));
    if(d_centroids_) CUDA_CALL(cudaFree(d_centroids_));
    if(d_csr_ptr_)   CUDA_CALL(cudaFree(d_csr_ptr_));
    if(d_csr_idx_)   CUDA_CALL(cudaFree(d_csr_idx_));
    CUBLAS_CALL( cublasDestroy(handle_) );
}

void IVFIndex::build(const float* data, size_t n){
    // k-means + CSR build
    // 先分配 idx
    h_csr_idx_ = (uint32_t*)malloc(sizeof(uint32_t)*n);
    cpu_kmeans(data, n, dim_, n_clusters_,
               h_centroids_, h_csr_ptr_, h_csr_idx_);
}

void IVFIndex::upload_to_gpu(const float* base_data, size_t n){
    // base vectors
    CUDA_CALL(cudaMalloc(&d_base_,sizeof(float)*n*dim_));
    CUDA_CALL(cudaMemcpy(d_base_,base_data,
             sizeof(float)*n*dim_,
             cudaMemcpyHostToDevice));
    // centroids
    CUDA_CALL(cudaMalloc(&d_centroids_,
             sizeof(float)*n_clusters_*dim_));
    CUDA_CALL(cudaMemcpy(d_centroids_,h_centroids_,
             sizeof(float)*n_clusters_*dim_,
             cudaMemcpyHostToDevice));
    // csr ptr
    CUDA_CALL(cudaMalloc(&d_csr_ptr_,
             sizeof(uint32_t)*(n_clusters_+1)));
    CUDA_CALL(cudaMemcpy(d_csr_ptr_,h_csr_ptr_,
             sizeof(uint32_t)*(n_clusters_+1),
             cudaMemcpyHostToDevice));
    // csr idx
    size_t tot = h_csr_ptr_[n_clusters_];
    CUDA_CALL(cudaMalloc(&d_csr_idx_,
             sizeof(uint32_t)*tot));
    CUDA_CALL(cudaMemcpy(d_csr_idx_,h_csr_idx_,
             sizeof(uint32_t)*tot,
             cudaMemcpyHostToDevice));
}

void IVFIndex::search_gpu_batch(
    const float* h_queries, size_t m, size_t k,
    uint32_t* h_out_ids, float* h_out_dists)
{
    int  D = int(dim_), C = int(n_clusters_), P = int(n_probe_);
    size_t L = m * P;

    // 1) H2D queries
    float* d_query=nullptr;
    CUDA_CALL(cudaMalloc(&d_query,sizeof(float)*D*m));
    CUDA_CALL(cudaMemcpy(d_query,h_queries,
             sizeof(float)*D*m,
             cudaMemcpyHostToDevice));

    // 2) alloc D1
    float* d_D1=nullptr;
    CUDA_CALL(cudaMalloc(&d_D1,sizeof(float)*C*m));

    // 3) 计算簇心距离矩阵
    dim3 blk(16,16), grd((C+blk.x-1)/blk.x,(m+blk.y-1)/blk.y);
    centroid_dist_kernel<<<grd,blk>>>(
        d_query, d_centroids_, d_D1, D, C, m);
    CUDA_CALL(cudaDeviceSynchronize());

    // 4) 拷回 D1，做每列 Top-P
    float* hD1 = (float*)malloc(sizeof(float)*C*m);
    CUDA_CALL(cudaMemcpy(hD1, d_D1,
             sizeof(float)*C*m,
             cudaMemcpyDeviceToHost));
    uint32_t* h_probe_cid = (uint32_t*)malloc(sizeof(uint32_t)*L);
    uint32_t* h_probe_qid = (uint32_t*)malloc(sizeof(uint32_t)*L);

    for(int q=0;q<m;q++){
        float* col = hD1 + q*C;
        // 生成 id 列表
        int* ids = (int*)malloc(sizeof(int)*C);
        for(int c=0;c<C;c++) ids[c]=c;
        // nth_element + sort 前 P
        std::nth_element(ids, ids+P, ids+C,
            [&](int a,int b){return col[a]<col[b];});
        std::sort(ids, ids+P,
            [&](int a,int b){return col[a]<col[b];});
        for(int p=0;p<P;p++){
            h_probe_cid[q*P + p] = ids[p];
            h_probe_qid[q*P + p] = q;
        }
        free(ids);
    }

    // 5) Host 分簇准备 Qlist
    uint32_t* Qcount = (uint32_t*)calloc(C,sizeof(uint32_t));
    uint32_t* Qlist  = (uint32_t*)malloc(sizeof(uint32_t)*C*m);
    for(size_t i=0;i<L;i++){
        uint32_t c = h_probe_cid[i];
        uint32_t q = h_probe_qid[i];
        uint32_t pos = Qcount[c]++;
        Qlist[c*m + pos] = q;
    }

    // 6) per-query min-heap for top-k
    std::vector< std::priority_queue<
        std::pair<float,uint32_t>>> heaps(m);

    // 7) 每簇 batch GEMM + 更新 heaps
    for(int c=0;c<C;c++){
        size_t mc = Qcount[c];
        if(mc==0) continue;
        // 7.1) GPU gather Qc
        float* d_Qc=nullptr;
        CUDA_CALL(cudaMalloc(&d_Qc,sizeof(float)*D*mc));
        for(size_t j=0;j<mc;j++){
            uint32_t qid = Qlist[c*m + j];
            CUDA_CALL(cudaMemcpy(
               d_Qc + j*D,
               d_query + qid*D,
               sizeof(float)*D,
               cudaMemcpyDeviceToDevice));
        }
        // 7.2) Bc 指针 & 大小
        uint32_t off = h_csr_ptr_[c];
        uint32_t rc  = h_csr_ptr_[c+1] - off;
        float*  d_Bc = d_base_ + size_t(off)*D;
        // 7.3) GEMM: rc×D * D×mc -> rc×mc
        float* d_Dc=nullptr;
        CUDA_CALL(cudaMalloc(&d_Dc,sizeof(float)*rc*mc));
        const float alpha=1.0f, beta=0.0f;
        CUBLAS_CALL( cublasSgemm(
            handle_,
            CUBLAS_OP_N,CUBLAS_OP_N,
            rc,int(mc),D,
            &alpha,
            d_Bc,rc,
            d_Qc,D,
            &beta,
            d_Dc,rc));
        // 7.4) 拷回 Dc
        float* hDc = (float*)malloc(sizeof(float)*rc*mc);
        CUDA_CALL(cudaMemcpy(
            hDc, d_Dc,
            sizeof(float)*rc*mc,
            cudaMemcpyDeviceToHost));
        // 7.5) 更新 heaps
        for(size_t j=0;j<mc;j++){
            uint32_t qid = Qlist[c*m + j];
            auto &hq = heaps[qid];
            for(uint32_t i2=0;i2<rc;i2++){
                float dist = hDc[i2 + j*rc];
                uint32_t gid = h_csr_idx_[off + i2];
                if(hq.size()<k || dist < hq.top().first){
                    hq.emplace(dist,gid);
                    if(hq.size()>k) hq.pop();
                }
            }
        }
        // free
        free(hDc);
        CUDA_CALL(cudaFree(d_Qc));
        CUDA_CALL(cudaFree(d_Dc));
    }

    // 8) 输出到 Host buffer
    for(size_t q=0;q<m;q++){
        auto &hq=heaps[q];
        size_t cnt=hq.size();
        for(size_t s=0;s<k;s++){
            if(s<cnt){
                auto pr=hq.top(); hq.pop();
                // 升序存放
                h_out_dists[q*k + (k-1-s)] = pr.first;
                h_out_ids  [q*k + (k-1-s)] = pr.second;
            } else {
                h_out_dists[q*k + (k-1-s)] = FLT_MAX;
                h_out_ids  [q*k + (k-1-s)] = 0xffffffffu;
            }
        }
    }

    // 9) cleanup
    free(hD1);
    free(h_probe_cid);
    free(h_probe_qid);
    free(Qcount);
    free(Qlist);
    CUDA_CALL(cudaFree(d_query));
    CUDA_CALL(cudaFree(d_D1));
}

