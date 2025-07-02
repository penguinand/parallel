#include "ivf_gpu.h"
#include <fstream>
#include <iostream>
#include <set>
#include <chrono>

struct SearchResult {
    float   recall;
    int64_t latency; // 单位 μs
};

template<typename T>
T* LoadData(const char* path, size_t& n, size_t& d) {
    std::ifstream fin(path, std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    fin.read((char*)data, sizeof(T)*n*d);
    fin.close();
    std::cerr<<"Loaded "<<path<<" #="<<n<<" dim="<<d<<"\n";
    return data;
}

int main(){
    size_t test_n, base_n, dim, gt_d;
    auto test_q = LoadData<float>(
        "DEEP100K.query.fbin", test_n, dim);
    auto gt     = LoadData<int>(
        "DEEP100K.gt.query.100k.top100.bin",
        test_n, gt_d);
    auto base   = LoadData<float>(
        "DEEP100K.base.100k.fbin", base_n, dim);

    size_t Q = std::min<size_t>(2000, test_n);
    const size_t K = 10;
    std::vector<SearchResult> results(Q);

    // Build & upload
    IVFIndex ivf(dim, 100, 5);
    ivf.build(base, base_n);
    ivf.upload_to_gpu(base, base_n);

    const size_t B = 128;
    uint32_t* out_ids   = new uint32_t[B*K];
    float*    out_dists = new float[B*K];

    for(size_t i=0; i<Q; i+=B){
        size_t mb = std::min(B, Q-i);
        auto t0 = std::chrono::high_resolution_clock::now();
        ivf.search_gpu_batch(
            test_q + i*dim, mb, K,
            out_ids, out_dists);
        auto t1 = std::chrono::high_resolution_clock::now();
        int64_t lat = std::chrono::duration_cast<
             std::chrono::microseconds>(t1-t0).count();
        for(size_t q=0; q<mb; q++){
            std::set<uint32_t> st;
            for(size_t t=0;t<K;t++){
                st.insert(gt[(i+q)*gt_d + t]);
            }
            size_t corr=0;
            for(size_t t=0;t<K;t++){
                if(st.count(out_ids[q*K+t])) corr++;
            }
            results[i+q] = { float(corr)/K, lat/mb };
        }
    }

    double avg_r=0; int64_t avg_l=0;
    for(size_t i=0;i<Q;i++){
        avg_r += results[i].recall;
        avg_l += results[i].latency;
    }
    avg_r/=Q; avg_l/=Q;
    std::cout<<"Average Recall: "<<avg_r<<"\n";
    std::cout<<"Average Latency: "<<avg_l<<" μs\n";

    delete[] test_q;
    delete[] gt;
    delete[] base;
    delete[] out_ids;
    delete[] out_dists;
    return 0;
}

