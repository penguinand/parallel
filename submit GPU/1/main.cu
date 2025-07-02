#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <chrono>
#include <limits>
#include <cstdlib>
#include <cstdint>
#include <cstring>

using namespace std;
using Clock = chrono::high_resolution_clock;

// CUDA 错误检查
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
      cerr << "CUDA Error " << __FILE__ << ":" << __LINE__                 \
           << " " << cudaGetErrorString(e) << endl;                        \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

// 从二进制文件读 n,d 及 n*d 数据
template<typename T>
T* LoadData(const string& path, size_t& n, size_t& d) {
  ifstream fin(path, ios::binary);
  if (!fin) { cerr<<"Cannot open "<<path<<endl; exit(-1); }
  uint32_t nn, dd;
  fin.read((char*)&nn,4);
  fin.read((char*)&dd,4);
  n=nn; d=dd;
  T* ptr = (T*)malloc(size_t(n)*d*sizeof(T));
  if(!ptr){ cerr<<"malloc failed "<<path<<endl; exit(-1); }
  fin.read((char*)ptr, size_t(n)*d*sizeof(T));
  fin.close();
  return ptr;
}

// kernel 计算 dist = base2 - 2*cross + query2
__global__ void compute_l2_kernel(
    const float* __restrict__ cross,
    const float* __restrict__ base2,
    const float* __restrict__ query2,
    float*       __restrict__ dist,
    int N, int Q)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<N && j<Q){
    float v = base2[i]
            - 2.0f*cross[size_t(i)+size_t(j)*N]
            + query2[j];
    dist[size_t(i)+size_t(j)*N] = v;
  }
}

int main(){
  // 配置区
  const string prefix  = "/home/s2312163/";
  const string f_base  = prefix+"DEEP100K.base.100k.fbin";
  const string f_query = prefix+"DEEP100K.query.fbin";
  const string f_gt    = prefix+"DEEP100K.gt.query.100k.top100.bin";
  const int    K       = 10;
  const int    QMAX    = 2000;

  // 1) Load
  size_t nb,db,nq,dq,dgt;
  float* base   = LoadData<float>(f_base,  nb, db);
  float* query0 = LoadData<float>(f_query, nq, dq);
  int*   gt0    = LoadData<int>  (f_gt,    nq, dgt);
  if(db!=dq){ cerr<<"Dim mismatch "<<db<<"!="<<dq<<endl; return -1; }
  int Q = int(min((size_t)QMAX, nq));

  // 2) CPU 预计算 Norm²
  vector<float> base2(nb), query2(Q);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<(int)nb;i++){
    float s=0;
    for(int d=0;d<(int)db;d++){
      float v=base[size_t(i)*db+d];
      s+=v*v;
    }
    base2[i]=s;
  }
  #pragma omp parallel for schedule(static)
  for(int i=0;i<Q;i++){
    float s=0;
    for(int d=0;d<(int)dq;d++){
      float v=query0[size_t(i)*dq+d];
      s+=v*v;
    }
    query2[i]=s;
  }

  // 3) GPU malloc + memcpy
  float *d_base, *d_query, *d_cross, *d_dist, *d_base2, *d_query2;
  size_t sz_base  = size_t(nb)*db  *sizeof(float);
  size_t sz_query = size_t(Q)*dq   *sizeof(float);
  size_t sz_cors  = size_t(nb)*Q   *sizeof(float);
  size_t sz_nb    = size_t(nb)     *sizeof(float);
  size_t sz_q     = size_t(Q)      *sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_base,   sz_base));
  CUDA_CHECK(cudaMalloc(&d_query,  sz_query));
  CUDA_CHECK(cudaMalloc(&d_cross,  sz_cors));
  CUDA_CHECK(cudaMalloc(&d_dist,   sz_cors));
  CUDA_CHECK(cudaMalloc(&d_base2,  sz_nb));
  CUDA_CHECK(cudaMalloc(&d_query2, sz_q));

  CUDA_CHECK(cudaMemcpy(d_base,   base,   sz_base,  cudaMemcpyHostToDevice));
  // query0 是 row-major Q×D，我们只用前 Q 条
  CUDA_CHECK(cudaMemcpy(d_query,  query0, sz_query, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_base2,  base2.data(),  sz_nb, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query2, query2.data(), sz_q,  cudaMemcpyHostToDevice));

  // cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Allocate host dist buffer
  vector<float> h_dist(size_t(nb)*Q);

  // 4) 【计时区间开始】从 GEMM
  auto T0 = Clock::now();

  // 4.1) cuBLAS GEMM: cross = base·query^T
  // 输入 row-major 都当作 (D×N)ᵀ 和 (D×Q)ᵀ
  const float alpha=1.0f, beta=0.0f;
  cublasSgemm(handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              nb, Q, db,
              &alpha,
              d_base,  db,
              d_query, db,
              &beta,
              d_cross, nb);

  // 4.2) L2 kernel
  dim3 block(16,16),
       grid((nb+15)/16,(Q+15)/16);
  compute_l2_kernel<<<grid,block>>>(d_cross,d_base2,d_query2,d_dist,(int)nb,Q);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4.3) D→H 拷回 dist 矩阵
  CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist,
      size_t(nb)*Q*sizeof(float),
      cudaMemcpyDeviceToHost));

  // 4.4) CPU 手写 Top-K
  vector< array<pair<float,int>,K> > all_res(Q);
  #pragma omp parallel for schedule(dynamic)
  for(int q=0;q<Q;q++){
    float* col = h_dist.data() + size_t(q)*nb;
    struct Item{ float d; int idx; };
    Item best[K];
    for(int i=0;i<K;i++){
      best[i].d = numeric_limits<float>::infinity();
      best[i].idx=-1;
    }
    int maxpos=0;
    for(int i=0;i<(int)nb;i++){
      float v=col[i];
      if(v>=best[maxpos].d) continue;
      best[maxpos].d=v;
      best[maxpos].idx=i;
      // 更新 maxpos
      float md=best[0].d; maxpos=0;
      for(int t=1;t<K;t++){
        if(best[t].d>md){ md=best[t].d; maxpos=t; }
      }
    }
    // 插入排序
    for(int a=1;a<K;a++){
      Item key=best[a]; int b=a-1;
      while(b>=0 && best[b].d>key.d){
        best[b+1]=best[b]; b--;
      }
      best[b+1]=key;
    }
    for(int t=0;t<K;t++){
      all_res[q][t]=make_pair(best[t].d,best[t].idx);
    }
  }

  // 4) 【计时区间结束】
  auto T1 = Clock::now();
  double elapsed = chrono::duration<double>(T1 - T0).count();

  // 5) Recall 评估
  double sum_rec=0;
  for(int q=0;q<Q;q++){
    int hit=0;
    for(int t=0;t<K;t++){
      int pred = all_res[q][t].second;
      for(int j=0;j<K;j++){
        if(pred == gt0[size_t(q)*dgt+j]){
          hit++; break;
        }
      }
    }
    sum_rec += double(hit)/K;
  }
  double avg_rec = sum_rec/Q;
  double avg_lat = elapsed*1e6/Q;  // μs/query

  // 输出
  cout<<"N="<<nb<<" D="<<db
      <<" Q="<<Q<<" K="<<K<<endl;
  cout<<"Average Recall:  "<<avg_rec<<endl;
  cout<<"Average Latency: "<<avg_lat<<" μs / query"<<endl;

  // cleanup
  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_base));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_cross));
  CUDA_CHECK(cudaFree(d_dist));
  CUDA_CHECK(cudaFree(d_base2));
  CUDA_CHECK(cudaFree(d_query2));
  free(base);
  free(query0);
  free(gt0);
  return 0;
}
