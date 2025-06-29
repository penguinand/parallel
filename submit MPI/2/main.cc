#include <mpi.h>
#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <set>
#include <chrono>
#include "pq_scan.h"

using namespace std;

// 从二进制文件里读取 n, d 以及紧接着的 n*d 个数据
template<typename T>
T* LoadData(const string& path, size_t& n, size_t& d) {
    ifstream fin(path, ios::binary);
    if (!fin) throw runtime_error("cannot open " + path);
    uint32_t nn, dd;
    fin.read((char*)&nn, 4);
    fin.read((char*)&dd, 4);
    n = nn; d = dd;
    T* data = (T*)malloc(size_t(n) * d * sizeof(T));
    fin.read((char*)data, size_t(n) * d * sizeof(T));
    fin.close();
    if (!data) throw bad_alloc();
    return data;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int  rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // 0) 常量配置
    const string prefix = "/anndata/";
    const size_t  QMAX   = 2000;
    const int     K      = 10;

    // 1) 由 rank=0 load 三个文件：query, gt, base
    size_t nq=0, d_q=0, d_gt=0, nb=0, d_b=0;
    float*  base     = nullptr;
    float*  queries0 = nullptr;
    int*    gt0      = nullptr;

    if (rank == 0) {
        // query
        queries0 = LoadData<float>(prefix + "DEEP100K.query.fbin", nq, d_q);
        // gt
        gt0      = LoadData<int>  (prefix + "DEEP100K.gt.query.100k.top100.bin", nq, d_gt);
        // base
        base     = LoadData<float>(prefix + "DEEP100K.base.100k.fbin", nb, d_b);
        if (d_b != d_q) {
            cerr << "base dimension != query dimension\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    // 2) 广播 nq, nb, d_q, d_gt
    MPI_Bcast(&nq,  1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb,  1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d_q, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d_gt,1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // 3) 所有 rank 分配 base 数组并且 Bcast
    if (rank != 0) {
        base = (float*)malloc(size_t(nb) * d_q * sizeof(float));
        if (!base) throw bad_alloc();
    }
    MPI_Bcast(base, int(nb*d_q), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 4) 准备把前 Q = min(QMAX, nq) 条 query Scatter 给各 rank
    size_t Q = min(QMAX, nq);
    // 先让每个 rank 知道它负责的 Q_local 数量
    vector<int> cnt(nprocs), disp(nprocs);
    size_t b = Q / nprocs, r = Q % nprocs;
    for (int i = 0; i < nprocs; i++) {
        cnt[i]  = int(b + (i < int(r) ? 1 : 0));
        disp[i] = (i == 0 ? 0 : disp[i-1] + cnt[i-1]);
    }
    int Q_local = cnt[rank];
    // rank=0 把 queries0 拆分；其他 rank 只需要分配 buffer
    vector<float> local_queries(size_t(Q_local) * d_q);
    {
        vector<int> scnt(nprocs), sdisp(nprocs);
        for (int i = 0; i < nprocs; i++) {
            scnt [i] = cnt[i] * int(d_q);
            sdisp[i] = disp[i] * int(d_q);
        }
        MPI_Scatterv(
            queries0, scnt.data(), sdisp.data(), MPI_FLOAT,
            local_queries.data(), Q_local*int(d_q), MPI_FLOAT,
            0, MPI_COMM_WORLD
        );
    }
    // gt_data 只要 rank=0 保留，用于最终 recall
    // queries0 之后可以 free
    if (rank == 0) {
        free(queries0);
    }

    // 5) 所有 rank 本地构建 PQIndex（训练+编码）
    PQIndex index(d_q);
    index.train(base, nb);
    index.encode(base, nb);

    // 6) barrier + timer
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // 7) 本 rank 在 local_queries 上做 batch_search
    vector<vector<pair<float,uint32_t>>> local_res(Q_local);
    batch_search(index,
                 local_queries.data(),
                 Q_local,
                 K,
                 local_res);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;

    // 8) 全局取 max elapsed（worst-case），用作平均 Latency 计算
    double max_elapsed = 0;
    MPI_Reduce(&local_elapsed,
               &max_elapsed,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);

    // 9) 本地 pack (Q_local×K) 结果到扁平数组
    vector<float>  send_d  (size_t(Q_local)*K);
    vector<int>    send_id (size_t(Q_local)*K);
    for (int i = 0; i < Q_local; i++) {
        for (int j = 0; j < K; j++) {
            send_d [i*K + j] = local_res[i][j].first;
            send_id[i*K + j] = int(local_res[i][j].second);
        }
    }

    // 10) 各 rank Gatherv 回 root
    vector<int> recv_cnt(nprocs), recv_disp(nprocs);
    for (int i = 0; i < nprocs; i++) {
        recv_cnt [i] = cnt[i] * K;
        recv_disp[i] = (i==0 ? 0 : recv_disp[i-1] + recv_cnt[i-1]);
    }
    vector<float> all_d;
    vector<int>   all_i;
    if (rank == 0) {
        all_d.resize(Q * K);
        all_i.resize(Q * K);
    }
    MPI_Gatherv(send_d.data(),
                Q_local*K, MPI_FLOAT,
                all_d.data(),
                recv_cnt.data(), recv_disp.data(),
                MPI_FLOAT,
                0, MPI_COMM_WORLD );

    MPI_Gatherv(send_id.data(),
                Q_local*K, MPI_INT,
                all_i.data(),
                recv_cnt.data(), recv_disp.data(),
                MPI_INT,
                0, MPI_COMM_WORLD );

    // 11) root 计算 Recall & 输出
    if (rank == 0) {
        // 重组 full results
        vector<vector<pair<float,uint32_t>>> results(Q);
        for (size_t i = 0; i < Q; i++) {
            results[i].resize(K);
            for (int j = 0; j < K; j++) {
                results[i][j] = make_pair(
                    all_d[i*K + j],
                    uint32_t(all_i[i*K + j])
                );
            }
        }
        // 计算 Recall
        double sum_rec = 0;
        for (size_t i = 0; i < Q; i++) {
            set<uint32_t> s;
            for (int j = 0; j < K; j++)
                s.insert(uint32_t(gt0[i*d_gt + j]));
            size_t hit = 0;
            for (auto &pr : results[i])
                if (s.count(pr.second)) ++hit;
            sum_rec += double(hit) / K;
        }
        double avg_rec = sum_rec / Q;
        double avg_lat = max_elapsed * 1e6 / Q;  // μs

        cout << "MPI ranks = " << nprocs << "\n";
        cout << "Average Recall:  "  << avg_rec << "\n";
        cout << "Average Latency: " << avg_lat << " μs\n";
    }

    // 12) cleanup
    if (rank == 0) {
        free(gt0);
    }
    free(base);
    MPI_Finalize();
    return 0;
}