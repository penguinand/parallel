#include <vector>
#include <queue>
#include <arm_neon.h>
#include <random>
#include <algorithm>
#include <omp.h>

constexpr size_t N_CLUSTERS = 100;
constexpr size_t N_PROBE = 5;
constexpr size_t IVF_KMEANS_ITER = 10;

class IVFIndex {
private:
    std::vector<std::vector<uint32_t>> clusters;
    std::vector<float> centroids;
    size_t dim;

    void kmeans_init(const float* data, size_t n) {
        std::mt19937 rng;
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        centroids.resize(N_CLUSTERS * dim);
        for (size_t c = 0; c < N_CLUSTERS; ++c) {
            memcpy(&centroids[c*dim], data + indices[c]*dim, dim*sizeof(float));
        }
    }

public:
    IVFIndex(size_t dim) : dim(dim) {}

    void build(const float* data, size_t n) {
        kmeans_init(data, n);
        
        std::vector<std::vector<uint32_t>> temp_clusters(N_CLUSTERS);

        for (size_t iter = 0; iter < IVF_KMEANS_ITER; ++iter) {
            // Per-thread temporary clusters to avoid critical section
            std::vector<std::vector<std::vector<uint32_t>>> thread_clusters(
                omp_get_max_threads(),
                std::vector<std::vector<uint32_t>>(N_CLUSTERS)
            );

            // Assign points to clusters
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                size_t best_c = 0;
                const float* vec = data + i*dim;
                
                for (size_t c = 0; c < N_CLUSTERS; ++c) {
                    float dist = 0.0f;
                    const float* centroid = &centroids[c*dim];
                    float32x4_t sum = vdupq_n_f32(0.0f);
                    size_t d = 0;

                    // Vectorized part
                    for (; d + 4 <= dim; d += 4) {
                        float32x4_t v = vld1q_f32(vec + d);
                        float32x4_t cb = vld1q_f32(centroid + d);
                        float32x4_t diff = vsubq_f32(v, cb);
                        sum = vmlaq_f32(sum, diff, diff);
                    }
                    // Scalar remainder
                    for (; d < dim; ++d) {
                        float diff = vec[d] - centroid[d];
                        dist += diff * diff;
                    }
                    dist += vaddvq_f32(sum);

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }

                int tid = omp_get_thread_num();
                thread_clusters[tid][best_c].push_back(i);
            }

            // Merge thread clusters
            for (auto& tc : thread_clusters) {
                for (size_t c = 0; c < N_CLUSTERS; ++c) {
                    auto& src = tc[c];
                    auto& dst = temp_clusters[c];
                    dst.insert(dst.end(), src.begin(), src.end());
                }
            }

            // Update centroids
            #pragma omp parallel for
            for (size_t c = 0; c < N_CLUSTERS; ++c) {
                if (temp_clusters[c].empty()) continue;
                
                std::vector<float> sum(dim, 0.0f);
                for (auto idx : temp_clusters[c]) {
                    const float* vec = data + idx*dim;
                    size_t d = 0;
                    // Vectorized accumulation
                    for (; d + 4 <= dim; d += 4) {
                        float32x4_t v = vld1q_f32(vec + d);
                        float32x4_t s = vld1q_f32(&sum[d]);
                        s = vaddq_f32(s, v);
                        vst1q_f32(&sum[d], s);
                    }
                    // Scalar remainder
                    for (; d < dim; ++d) {
                        sum[d] += vec[d];
                    }
                }
                
                float* centroid = &centroids[c*dim];
                float inv_size = 1.0f / temp_clusters[c].size();
                size_t d = 0;
                // Vectorized scaling
                for (; d + 4 <= dim; d += 4) {
                    float32x4_t s = vld1q_f32(&sum[d]);
                    s = vmulq_n_f32(s, inv_size);
                    vst1q_f32(centroid + d, s);
                }
                // Scalar remainder
                for (; d < dim; ++d) {
                    centroid[d] = sum[d] * inv_size;
                }
            }

            // Clear for next iteration
            if (iter != IVF_KMEANS_ITER - 1) {
                for (auto& cluster : temp_clusters) cluster.clear();
            }
        }

        clusters = std::move(temp_clusters);
    }

    std::vector<size_t> search_clusters(const float* query, size_t n_probe) const {
        std::priority_queue<std::pair<float, size_t>> pq;
        for (size_t c = 0; c < N_CLUSTERS; ++c) {
            float dist = 0.0f;
            const float* centroid = &centroids[c*dim];
            float32x4_t sum = vdupq_n_f32(0.0f);
            size_t d = 0;

            // Vectorized part
            for (; d + 4 <= dim; d += 4) {
                float32x4_t q = vld1q_f32(query + d);
                float32x4_t cb = vld1q_f32(centroid + d);
                float32x4_t diff = vsubq_f32(q, cb);
                sum = vmlaq_f32(sum, diff, diff);
            }
            // Scalar remainder
            for (; d < dim; ++d) {
                float diff = query[d] - centroid[d];
                dist += diff * diff;
            }
            dist += vaddvq_f32(sum);

            pq.emplace(dist, c);
            if (pq.size() > n_probe) pq.pop();
        }

        std::vector<size_t> res;
        while (!pq.empty()) {
            res.push_back(pq.top().second);
            pq.pop();
        }
        return res;
    }

    const std::vector<uint32_t>& get_cluster(size_t cid) const {
        return clusters[cid];
    }
};

std::priority_queue<std::pair<float, uint32_t>> ivf_search(
    const IVFIndex& index,
    const float* base,
    const float* query,
    size_t vecdim,
    size_t k)
{
    auto cluster_ids = index.search_clusters(query, N_PROBE);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_queues(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < cluster_ids.size(); ++i) {
        auto cid = cluster_ids[i];
        const auto& cluster = index.get_cluster(cid);
        int tid = omp_get_thread_num();
        auto& local_q = local_queues[tid];
        
        for (uint32_t idx : cluster) {
            const float* vec = base + idx * vecdim;
            float dis = 0.0f;
            float32x4_t sum = vdupq_n_f32(0.0f);
            size_t d = 0;

            // Vectorized part
            for (; d + 4 <= vecdim; d += 4) {
                float32x4_t q = vld1q_f32(query + d);
                float32x4_t v = vld1q_f32(vec + d);
                float32x4_t diff = vsubq_f32(q, v);
                sum = vmlaq_f32(sum, diff, diff);
            }
            // Scalar remainder
            for (; d < vecdim; ++d) {
                float diff = query[d] - vec[d];
                dis += diff * diff;
            }
            dis += vaddvq_f32(sum);

            if (local_q.size() < k || dis < local_q.top().first) {
                local_q.emplace(dis, idx);
                if (local_q.size() > k) local_q.pop();
            }
        }
    }

    // Merge results
    std::priority_queue<std::pair<float, uint32_t>> final_q;
    for (auto& q : local_queues) {
        while (!q.empty()) {
            final_q.push(q.top());
            q.pop();
            if (final_q.size() > k) final_q.pop();
        }
    }
    return final_q;
}
