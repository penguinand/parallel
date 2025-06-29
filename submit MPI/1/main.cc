// main_mpi.cc
#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <omp.h>
#include "scan.h"

using namespace std;
using Clock = chrono::high_resolution_clock;

// 读二进制 .fbin，格式：int32 n, int32 d, 接着 n*d 个 T
template<typename T>
T* LoadData(const string& path, size_t& n, size_t& d) {
    ifstream fin(path, ios::binary);
    if (!fin) throw runtime_error("cannot open " + path);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    fin.read((char*)data, sizeof(T) * n * d);
    fin.close();
    return data;
}

#pragma pack(push,1)
struct Hit {
    float    dist;
    uint32_t id;
};
#pragma pack(pop)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const string PREFIX = "/anndata/";
    size_t nq=0, nb=0, dq=0, dgt=0;
    float* base_all = nullptr;
    float* queries_all = nullptr;
    int*   gt_all      = nullptr;
    if (rank == 0) {
        // base
        base_all = LoadData<float>(PREFIX + "DEEP100K.base.100k.fbin", nb, dq);
        // queries & ground-truth
        queries_all = LoadData<float>(PREFIX + "DEEP100K.query.fbin",
                                      nq, dq);
        gt_all      = LoadData<int>(
                         PREFIX + "DEEP100K.gt.query.100k.top100.bin",
                         nq, dgt);
    }
    size_t Q = (rank==0? min<size_t>(2000, nq) : 0);
    size_t K = 10;

    int nb_i, dq_i, Q_i, K_i, dgt_i;
    if (rank==0) {
        nb_i = int(nb);
        dq_i = int(dq);
        Q_i  = int(Q);
        K_i  = int(K);
        dgt_i= int(dgt);
    }
    MPI_Bcast(&nb_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dq_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Q_i,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K_i,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dgt_i,1, MPI_INT, 0, MPI_COMM_WORLD);
    nb = nb_i; dq = dq_i; Q = Q_i; K = K_i; dgt = dgt_i;

    if (rank != 0) {
        base_all = new float[ size_t(nb) * dq ];
    }
    MPI_Bcast(base_all,
              nb * dq,
              MPI_FLOAT,
              0,
              MPI_COMM_WORLD);

    omp_set_dynamic(0);
    omp_set_num_threads(1);

    PQIndex index(dq);
    index.train(base_all, nb);
    index.encode(base_all, nb);

    vector<int> qcnt(world), qdisp(world);
    int qbase = Q / world, rem = Q % world;
    {
        int offset = 0;
        for (int r = 0; r < world; r++) {
            qcnt[r] = (r < rem ? qbase+1 : qbase);
            qdisp[r] = offset;
            offset += qcnt[r];
        }
    }
    vector<int> sendCounts(world), sendDispls(world);
    for (int r = 0; r < world; r++) {
        sendCounts[r] = qcnt[r] * dq;
        sendDispls[r] = qdisp[r] * dq;
    }
    int myQ = qcnt[rank];
    float* my_queries = new float[ size_t(myQ) * dq ];

    MPI_Scatterv( queries_all,
                  sendCounts.data(),
                  sendDispls.data(),
                  MPI_FLOAT,
                  my_queries,
                  myQ * dq,
                  MPI_FLOAT,
                  0,
                  MPI_COMM_WORLD );
    if (rank==0) {
        delete[] queries_all;
        queries_all = nullptr;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = Clock::now();

    vector<Hit> my_hits(size_t(myQ) * K);
    omp_set_dynamic(0);
    omp_set_num_threads( omp_get_max_threads() );

    vector<vector<pair<float,uint32_t>>> local_res(myQ);
    batch_search(index,
                 my_queries,
                 myQ,
                 K,
                 local_res);

    for (int i = 0; i < myQ; i++) {
        for (int j = 0; j < K; j++) {
            my_hits[ size_t(i)*K + j ].dist = local_res[i][j].first;
            my_hits[ size_t(i)*K + j ].id   = local_res[i][j].second;
        }
    }
    delete[] my_queries;
    local_res.clear();

    vector<int> sendBytes(world), recvBytes(world), recvDispls(world);
    for (int r = 0; r < world; r++) {
        sendBytes[r] = qcnt[r] * K * sizeof(Hit);
    }
    if (rank == 0) {
        recvBytes = sendBytes;
        recvDispls[0] = 0;
        for (int r = 1; r < world; r++) {
            recvDispls[r] = recvDispls[r-1] + recvBytes[r-1];
        }
    }
    Hit* all_hits = nullptr;
    if (rank == 0) {
        all_hits = (Hit*)malloc( size_t(Q) * K * sizeof(Hit) );
    }

    MPI_Gatherv( my_hits.data(),
                 sendBytes[rank],
                 MPI_BYTE,
                 all_hits,
                 recvBytes.data(),
                 recvDispls.data(),
                 MPI_BYTE,
                 0,
                 MPI_COMM_WORLD );

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = Clock::now();

    if (rank == 0) {
        double total_us = chrono::duration<double, micro>(t1 - t0).count();
        double avg_lat  = total_us / Q;

        double sum_rec = 0;
        for (int qi = 0; qi < (int)Q; qi++) {
            // 构造 GT set
            set<uint32_t> gtset;
            for (int j = 0; j < (int)K; j++) {
                gtset.insert( uint32_t(gt_all[ size_t(qi)*dgt + j ]) );
            }
            // 读 top-K
            int hit = 0;
            Hit* hits = all_hits + size_t(qi)*K;
            for (int j = 0; j < (int)K; j++) {
                if (gtset.count(hits[j].id)) ++hit;
            }
            sum_rec += double(hit) / K;
        }
        double avg_rec = sum_rec / Q;
        cout << "Average Recall:  " << avg_rec  << "\n";
        cout << "Average Latency: " << avg_lat << " us\n";

        // cleanup
        free(all_hits);
        delete[] gt_all;
        delete[] base_all;
    }

    MPI_Finalize();
    return 0;
}
