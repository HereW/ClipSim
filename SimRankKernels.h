#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "GraphMatrix.h"
#include <iostream>

using namespace std;

#define MAX_THREAD   512
#define MAX_BLOCK    512
#define TOTAL_THREAD 262144
#define TOTAL_WARPS  2048
// #define TOTAL_WARPS  8192
#define WARP_SIZE    32
#define RW_THRESHOLD 1024
// #define JUMP_NUMBER  20

// vector addtion, Y = a * X + Y. 
__global__ void parallel_addition(int n, float a, Dense_vector<float> X, Dense_vector<float> Y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Y.entries[i] = a * X.entries[i] + Y.entries[i];
    }
}

__global__ void diag_matrix_dense_vector_multiplication(float* dDiag_values, Dense_vector<float> denseX,
                                            Dense_vector<float> denseY, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < denseX.num_rows) {
        denseY.entries[i] = alpha * dDiag_values[i] * denseX.entries[i];
    }
}

__global__ void find_end_vert_per_thread(Dense_vector<int> endVertIdx, int num_vertices) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = id / WARP_SIZE;
    endVertIdx.entries[wid] = (num_vertices / TOTAL_WARPS - 1) * TOTAL_WARPS + wid;
    int residue = num_vertices % TOTAL_WARPS;
    if (wid < residue)
        endVertIdx.entries[wid] += TOTAL_WARPS;
}

// __global__ void decide_each_vert_portion_walk_num(int* portion_walk, int* each_portion_walk_num, Dense_vector<float> PPR_values, long total_walk_num, float PPR_norm2) {
__global__ void decide_each_vert_portion_walk_num(int* portion_walk, int* each_portion_walk_num, Dense_vector<float> PPR_values, long total_walk_num) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n = PPR_values.num_rows;
    int q = (n + TOTAL_THREAD - 1) / TOTAL_THREAD;

    for (int qi = 0; qi < q; qi++) {
        int qid = qi * TOTAL_THREAD + id;
        if (qid < n) {
            // long walk_num_k = (long)(total_walk_num * PPR_values.entries[qid]) + 1;
            long walk_num_k = (long)(total_walk_num * PPR_values.entries[qid] * PPR_values.entries[qid] / (PPR_values.entries[qid] + 0.1)) + 1;
            // printf("qid:%d, walk_num_k:%ld\n", qid, walk_num_k);
            portion_walk[qid] = (walk_num_k + RW_THRESHOLD - 1) / RW_THRESHOLD * WARP_SIZE;
            each_portion_walk_num[qid] = RW_THRESHOLD / WARP_SIZE;
            if (walk_num_k < RW_THRESHOLD) {
                // each_portion_walk_num[qid] = WARP_SIZE;
                // portion_walk[qid] = RW_THRESHOLD / WARP_SIZE;
                each_portion_walk_num[qid] = 0;
                portion_walk[qid] = 0;
            }
        }
    }
}

__global__ void calculate_matrix_D(float* diag_values, float c_value, Dense_vector<int> indegrees, unsigned int* meets_count, int* portion_walk, int* each_portion_walk_num) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n = indegrees.num_rows;
    int q = (n + TOTAL_THREAD - 1) / TOTAL_THREAD;

    for (int qi = 0; qi < q; qi++) {
        int qid = qi * TOTAL_THREAD + id;
        if (qid < n) {
            int walk_num_qid = portion_walk[qid] * each_portion_walk_num[qid];
            if (indegrees.entries[qid] == 0)
                diag_values[qid] = 1.0;
            else {
                if (walk_num_qid > 0)
                    diag_values[qid] = 1.0 - (c_value / indegrees.entries[qid]) - (c_value * meets_count[qid] / walk_num_qid);
                else
                    diag_values[qid] = 1.0 - (c_value / indegrees.entries[qid]);
            }
            // printf("qid:%d, D:%2.4f\n", qid, diag_values[qid]);
        }
    }
}

__global__ void matrix_P_construction(int* indegrees, int* columns, float* values, int m) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int q = (m + TOTAL_THREAD - 1) / TOTAL_THREAD;

    for (int qi = 0; qi < q; qi++) {
        int qid = qi * TOTAL_THREAD + id;
        if (qid < m) {
            values[qid] = 1.0 / indegrees[columns[qid]];
        }
    }
}

__device__ void single_vertex_random_walk_v2(curandState &r_state, curandState &wid_state, Sparse_CSR dA, int node, float c_v, int portion_walk_num, unsigned int& meet_num) {
    
    int u_newNode = node, v_newNode = node, u_nextNode, v_nextNode;

    for (int j = 0; j < portion_walk_num; j++) {
        int length = dA.csrOffsets[node + 1] - dA.csrOffsets[node];
        if (length > 0) {
            u_newNode = dA.columns[dA.csrOffsets[node] + (curand(&r_state) % length)];
            v_newNode = dA.columns[dA.csrOffsets[node] + (curand(&r_state) % length)];
        }
        
        int indicator = 1;

        if (u_newNode == v_newNode) {
            if (indicator == 1) {
                indicator = 0;
            }
        } 

        while (abs(curand_uniform_double(&wid_state)) < c_v) {
            if (indicator == 1) {
                int length = dA.csrOffsets[u_newNode + 1] - dA.csrOffsets[u_newNode];
                if (length == 0) {
                    if (indicator == 1) {
                        indicator = 0;
                    }
                }
                if (indicator == 1) {
                    int r = curand(&r_state) % length;
                    u_nextNode = dA.columns[dA.csrOffsets[u_newNode] + r];
                    length = dA.csrOffsets[v_newNode + 1] - dA.csrOffsets[v_newNode];
                    if (length == 0) {
                        if (indicator == 1) {
                            indicator = 0;
                        }
                    }
                    if (indicator == 1) {
                        r = curand(&r_state) % length;
                        v_nextNode = dA.columns[dA.csrOffsets[v_newNode] + r];
                        if (u_nextNode == v_nextNode) {
                            meet_num += 1;
                            if (indicator == 1) {
                                indicator = 0;
                            }
                        }
                        u_newNode = u_nextNode;
                        v_newNode = v_nextNode;
                    }
                }
            }
            // __syncwarp();
        }
    }
}

__global__ void parallel_random_walk_v3_2 (long rand, Sparse_CSR dM, Dense_vector<int> end_vert_idx, int* portion_walk, 
                    int* each_portion_walk_num, float c_v, unsigned int* meet_count, int jump_number, bool dual_seeds) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = id / WARP_SIZE;

    int cur_k = wid;

    curandState state;
    curandState w_state;

    long seed = rand;
    curand_init(seed, id, 0, &state);
    if (dual_seeds)
        curand_init(seed, wid, 0, &w_state);
    else
        curand_init(seed, id, 0, &w_state);

    while ((cur_k < end_vert_idx.entries[wid]) && (cur_k < dM.num_rows)) {
        while (portion_walk[cur_k] > 0) {
            atomicAdd(&(portion_walk[cur_k]), -1);
            unsigned int meet_count_k = 0;
            single_vertex_random_walk_v2(state, w_state, dM, cur_k, c_v, each_portion_walk_num[cur_k], meet_count_k);
            atomicAdd(&(meet_count[cur_k]), meet_count_k);
        }
        cur_k += TOTAL_WARPS;
    }

    // if ((id % WARP_SIZE == 0) && (cur_k < dM.num_rows)) {
    //     printf("cur_k:%d, meet_count:%d\n", cur_k, meet_count[cur_k]);
    // }

    int jump_count = 0;
    int jump_pos;
    int work_vert;

    while (jump_count < jump_number) {

        jump_pos = curand(&w_state) % TOTAL_WARPS;
        work_vert = end_vert_idx.entries[jump_pos];
        while ((work_vert >= TOTAL_WARPS) && (portion_walk[work_vert] <= 0))
            work_vert -= TOTAL_WARPS;

        if (portion_walk[work_vert] > 0) {
            atomicAdd(&(portion_walk[work_vert]), -1);
            unsigned int meet_count_v = 0;
            single_vertex_random_walk_v2(state, w_state, dM, work_vert, c_v, each_portion_walk_num[work_vert], meet_count_v);
            atomicAdd(&(meet_count[work_vert]), meet_count_v);
        }

        jump_count += 1;
    }
}