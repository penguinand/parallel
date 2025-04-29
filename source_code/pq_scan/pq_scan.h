#include <vector>
#include <cstring>
#include <random>
#include <numeric>
#include <limits>
#include <arm_neon.h>

constexpr size_t SUBSPACE_NUM = 4;
constexpr size_t SUBSPACE_DIM = 32;
constexpr size_t CENTROID_NUM = 256;
constexpr size_t KMEANS_ITER = 10;

class PQIndex {
private:
    std::vector<uint8_t> codes;
    float* codebook[SUBSPACE_NUM];
    size_t dim;

    void kmeans_init(const float* data, size_t n, size_t subspace, float* centroids, std::mt19937& rng) {
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        memcpy(centroids, data + indices[0] * SUBSPACE_DIM, SUBSPACE_DIM * sizeof(float));

        std::vector<float> min_dists(n, std::numeric_limits<float>::max());
        for (size_t c = 1; c < CENTROID_NUM; ++c) {
            // Compute distances to existing centroids
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                const float* vec = data + i * SUBSPACE_DIM;
                float min_dist = std::numeric_limits<float>::max();
                for (size_t pc = 0; pc < c; ++pc) {
                    const float* cent = centroids + pc * SUBSPACE_DIM;
                    float32x4_t sum = vdupq_n_f32(0.0f);
                    for (size_t d = 0; d < SUBSPACE_DIM; d += 4) {
                        float32x4_t v = vld1q_f32(vec + d);
                        float32x4_t c = vld1q_f32(cent + d);
                        float32x4_t diff = vsubq_f32(v, c);
                        sum = vmlaq_f32(sum, diff, diff);
                    }
                    float dist = vaddvq_f32(sum);
                    if (dist < min_dist) min_dist = dist;
                }
                min_dists[i] = min_dist;
            }

            // Build cumulative distribution
            std::vector<float> cum_dists(n);
            std::partial_sum(min_dists.begin(), min_dists.end(), cum_dists.begin());
            std::uniform_real_distribution<float> dist(0.0f, cum_dists.back());
            float r = dist(rng);

            // Select next centroid
            size_t selected = std::lower_bound(cum_dists.begin(), cum_dists.end(), r) - cum_dists.begin();
            selected = std::min(selected, n - 1);
            memcpy(centroids + c * SUBSPACE_DIM, data + selected * SUBSPACE_DIM, SUBSPACE_DIM * sizeof(float));
        }
    }

public:
    PQIndex(size_t dim) : dim(dim) {
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            codebook[s] = (float*)aligned_alloc(16, CENTROID_NUM * SUBSPACE_DIM * sizeof(float));
        }
    }

    ~PQIndex() {
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            free(codebook[s]);
        }
    }

    void train(float* data, size_t n) {
        #pragma omp parallel for
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            float* subspace_data = new float[n * SUBSPACE_DIM];
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                memcpy(subspace_data + i * SUBSPACE_DIM, data + i * dim + s * SUBSPACE_DIM, SUBSPACE_DIM * sizeof(float));
            }

            std::mt19937 rng(s); // Seed with subspace for reproducibility
            kmeans_init(subspace_data, n, s, codebook[s], rng);

            std::vector<size_t> assignments(n);
            for (size_t iter = 0; iter < KMEANS_ITER; ++iter) {
                // Assignment step
                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    const float* vec = subspace_data + i * SUBSPACE_DIM;
                    float min_dist = std::numeric_limits<float>::max();
                    size_t best_c = 0;
                    for (size_t c = 0; c < CENTROID_NUM; ++c) {
                        const float* cent = codebook[s] + c * SUBSPACE_DIM;
                        float32x4_t sum = vdupq_n_f32(0.0f);
                        for (size_t d = 0; d < SUBSPACE_DIM; d += 4) {
                            float32x4_t v = vld1q_f32(vec + d);
                            float32x4_t cb = vld1q_f32(cent + d);
                            float32x4_t diff = vsubq_f32(v, cb);
                            sum = vmlaq_f32(sum, diff, diff);
                        }
                        float dist = vaddvq_f32(sum);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_c = c;
                        }
                    }
                    assignments[i] = best_c;
                }

                // Update step with efficient accumulation
                std::vector<float> sum_total(CENTROID_NUM * SUBSPACE_DIM, 0.0f);
                std::vector<size_t> count_total(CENTROID_NUM, 0);

                #pragma omp parallel
                {
                    std::vector<float> sum_local(CENTROID_NUM * SUBSPACE_DIM, 0.0f);
                    std::vector<size_t> count_local(CENTROID_NUM, 0);

                    #pragma omp for nowait
                    for (size_t i = 0; i < n; ++i) {
                        size_t c = assignments[i];
                        const float* vec = subspace_data + i * SUBSPACE_DIM;
                        for (size_t d = 0; d < SUBSPACE_DIM; ++d) {
                            sum_local[c * SUBSPACE_DIM + d] += vec[d];
                        }
                        count_local[c]++;
                    }

                    #pragma omp critical
                    {
                        for (size_t c = 0; c < CENTROID_NUM; ++c) {
                            count_total[c] += count_local[c];
                            for (size_t d = 0; d < SUBSPACE_DIM; ++d) {
                                sum_total[c * SUBSPACE_DIM + d] += sum_local[c * SUBSPACE_DIM + d];
                            }
                        }
                    }
                }

                // Handle empty clusters and update centroids
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<size_t> idx_dist(0, n - 1);

                #pragma omp parallel for
                for (size_t c = 0; c < CENTROID_NUM; ++c) {
                    if (count_total[c] == 0) {
                        size_t selected = idx_dist(gen);
                        memcpy(codebook[s] + c * SUBSPACE_DIM,
                               subspace_data + selected * SUBSPACE_DIM,
                               SUBSPACE_DIM * sizeof(float));
                    } else {
                        float* cent = codebook[s] + c * SUBSPACE_DIM;
                        for (size_t d = 0; d < SUBSPACE_DIM; ++d) {
                            cent[d] = sum_total[c * SUBSPACE_DIM + d] / count_total[c];
                        }
                    }
                }
            }
            delete[] subspace_data;
        }
    }

void encode(float* data, size_t n) {
        codes.resize(n * SUBSPACE_NUM);
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
                const float* vec = data + i*dim + s*SUBSPACE_DIM;
                float min_dist = std::numeric_limits<float>::max();
                uint8_t best_c = 0;
                for (size_t c = 0; c < CENTROID_NUM; ++c) {
                    float dist = 0;
                    const float* centroid = codebook[s] + c*SUBSPACE_DIM;
                    float32x4_t sum = vdupq_n_f32(0.0f);
                    for (size_t d = 0; d < SUBSPACE_DIM; d += 4) {
                        float32x4_t v = vld1q_f32(vec + d);
                        float32x4_t cb = vld1q_f32(centroid + d);
                        float32x4_t diff = vsubq_f32(v, cb);
                        sum = vmlaq_f32(sum, diff, diff);
                    }
                    dist = vaddvq_f32(sum);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                codes[i*SUBSPACE_NUM + s] = best_c;
            }
        }
    }

    void precompute_LUT(const float* query, float* lut) const {
        #pragma omp parallel for collapse(2)
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            for (size_t c = 0; c < CENTROID_NUM; ++c) {
                const float* centroid = codebook[s] + c*SUBSPACE_DIM;
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (size_t d = 0; d < SUBSPACE_DIM; d += 4) {
                    float32x4_t q = vld1q_f32(query + s*SUBSPACE_DIM + d);
                    float32x4_t cb = vld1q_f32(centroid + d);
                    float32x4_t diff = vsubq_f32(q, cb);
                    sum = vmlaq_f32(sum, diff, diff);
                }
                lut[s*CENTROID_NUM + c] = vaddvq_f32(sum);
            }
        }
    }

    const uint8_t* get_codes() const { return codes.data(); }
};

std::priority_queue<std::pair<float, uint32_t>> pq_search(
    const PQIndex& index,
    const float* query,
    size_t base_number,
    size_t k)
{
    thread_local float lut[SUBSPACE_NUM * CENTROID_NUM];
    index.precompute_LUT(query, lut);

    const uint8_t* codes = index.get_codes();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_queues(omp_get_max_threads());

    #pragma omp parallel for
    for (size_t i = 0; i < base_number; ++i) {
        float dis = 0;
        const uint8_t* code = codes + i * SUBSPACE_NUM;
        for (size_t s = 0; s < SUBSPACE_NUM; ++s) {
            dis += lut[s * CENTROID_NUM + code[s]];
        }

        auto& q = local_queues[omp_get_thread_num()];
        if (q.size() < k) {
            q.emplace(dis, i);
        } else if (dis < q.top().first) {
            q.emplace(dis, i);
            if (q.size() > k) q.pop();
        }
    }

    // 合并各线程局部队列
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