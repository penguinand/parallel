#ifndef IVF_GPU_H_
#define IVF_GPU_H_

#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

// IVFIndex: CPU 上 build，GPU 上做矩阵乘+批量检索
class IVFIndex {
public:
    IVFIndex(size_t dim, size_t n_clusters=100, size_t n_probe=5);
    ~IVFIndex();

    // 在 CPU 上对 data[n*dim] 做 k-means
    void build(const float* data, size_t n);

    // build 完成后把 base_data[n*dim] + centroids + CSR 拷到 GPU
    void upload_to_gpu(const float* base_data, size_t n);

    // 批量 search：h_queries[m*dim]，找 top-k
    // 输出 row-major：results_ids[m*k], results_dists[m*k]
    void search_gpu_batch(
        const float* h_queries, size_t m, size_t k,
        uint32_t* results_ids, float* results_dists);

private:
    size_t dim_, n_clusters_, n_probe_;
    // CPU 数据
    float*    h_centroids_;   // [n_clusters_*dim_]
    uint32_t* h_csr_ptr_;     // [n_clusters_+1]
    uint32_t* h_csr_idx_;     // [总点数]

    // GPU 数据
    float*    d_base_       = nullptr;  // [n*dim]
    float*    d_centroids_  = nullptr;  // [n_clusters*dim]
    uint32_t* d_csr_ptr_    = nullptr;  // [n_clusters+1]
    uint32_t* d_csr_idx_    = nullptr;  // [总点数]

    cublasHandle_t handle_;

    // 禁止拷贝
    IVFIndex(const IVFIndex&) = delete;
    IVFIndex& operator=(const IVFIndex&) = delete;
};

#endif // IVF_GPU_H_

