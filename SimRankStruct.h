#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cublas_v2.h>
#include <cusparse.h>         
#include <stdio.h>            
#include <stdlib.h>           
#include <typeinfo>
#include <iostream>
#include <iomanip> 
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h" 
#include <curand_kernel.h>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "SimRankKernels.h"
#include "GraphMatrix.h"
#include "load_data.h"

using namespace std;

#define MAX_THREAD   512
#define MAX_BLOCK    512
#define TOTAL_THREAD 262144
#define TOTAL_WARPS  2048
// #define TOTAL_WARPS  8192

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class SimRankStruct {
public:
    string filelabel;
    int vert;
    int edges; 
    float sqrtC;
    float c;
    float eps;
    float delta;
    float avg_time;
    long walk_num;
    int L;
    Sparse_COO hP_COO;
    int* dP_csrOffsets, * dP_columns, * dP_COO_rows;
    int* dP_indegrees;
    float* dP_values;
    Sparse_CSR dP;
    Sparse_CSR dP_T;
    float* hVecS;
    float* dDiag_values;
    float* dVecMid_ent;
    float* dTmpVec_ent;
    float* dVecS_ent;
    Dense_vector<float>* dPPR_all;
    Dense_vector<float> dVecMid;
    Dense_vector<float> dTmpVecS;
    int P_num_nonzeros;
    int P_num_rows;
    int P_num_cols;
    int vecS_num_rows;
    int jump_num;
    bool dual_flag;
    
    // The multiplication of sparse matrix and dense vector. 
    void sparse_csr_matrix_dense_vector_multiplication(cusparseHandle_t &handle, Sparse_CSR sparseA, Dense_vector<float> denseX, Dense_vector<float> denseY,
        cusparseOperation_t trans, float alpha = 1.0f) {
        long long            Y_num_rows = sparseA.num_rows;
        float                beta = 0.0f;
        //--------------------------------------------------------------------------
        // CUSPARSE APIs
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        void* dBuffer = NULL;
        size_t               bufferSize = 0;
        cusparseCreate(&handle);
        // Create sparse matrix A in CSR format
        cusparseCreateCsr(&matA, sparseA.num_rows, sparseA.num_cols, sparseA.num_nonzeros,
            sparseA.csrOffsets, sparseA.columns, sparseA.values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        // Create dense vector X
        cusparseCreateDnVec(&vecX, denseX.num_rows, denseX.entries, CUDA_R_32F);
        // Create dense vector Y
        cusparseCreateDnVec(&vecY, Y_num_rows, denseY.entries, CUDA_R_32F);
        // allocate an external buffer if needed
        cusparseSpMV_bufferSize(handle, trans,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);
        // execute SpMV
        cusparseSpMV(handle, trans,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, dBuffer);
        // destroy matrix/vector descriptors
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        //--------------------------------------------------------------------------
        // device memory deallocation
        cudaFree(dBuffer);
        //--------------------------------------------------------------------------
        //return denseY;
    }

    // The convertion between Sparse_CSR and SpMat, Dense_Vec and DnVec. 
    void sparse_csr_matrix_dense_vector_multiplication_setting(cusparseHandle_t &handle, Sparse_CSR sparseA, 
        Dense_vector<float> denseX, Dense_vector<float> denseY, 
        cusparseSpMatDescr_t &matA, cusparseDnVecDescr_t &vecX, cusparseDnVecDescr_t &vecY, void* dBuffer, 
        cusparseOperation_t trans, float alpha = 1.0f) {
        //--------------------------------------------------------------------------
        size_t               bufferSize = 0;
        float                beta = 0.0f;
        //--------------------------------------------------------------------------
        // Create sparse matrix A in CSR format
        cusparseCreateCsr(&matA, sparseA.num_rows, sparseA.num_cols, sparseA.num_nonzeros,
            sparseA.csrOffsets, sparseA.columns, sparseA.values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        // Create dense vector X
        cusparseCreateDnVec(&vecX, denseX.num_rows, denseX.entries, CUDA_R_32F);
        // Create dense vector Y
        cusparseCreateDnVec(&vecY, sparseA.num_rows, denseY.entries, CUDA_R_32F);
        // allocate an external buffer if needed
        cusparseSpMV_bufferSize(handle, trans,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);
    }

    void sparse_csr_matrix_dense_vector_multiplication_MAT_setting(cusparseHandle_t &handle, 
        Sparse_CSR sparseA, cusparseSpMatDescr_t &matA) {
        //--------------------------------------------------------------------------
        // Create sparse matrix A in CSR format
        cusparseCreateCsr(&matA, sparseA.num_rows, sparseA.num_cols, sparseA.num_nonzeros,
            sparseA.csrOffsets, sparseA.columns, sparseA.values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    }

    void sparse_csr_matrix_dense_vector_multiplication_VEC_setting(cusparseHandle_t &handle, 
        Dense_vector<float> denseX, Dense_vector<float> denseY, 
        cusparseDnVecDescr_t &vecX, cusparseDnVecDescr_t &vecY) {
        //-------------------------------------------------------------------------
        // Create dense vector X
        cusparseCreateDnVec(&vecX, denseX.num_rows, denseX.entries, CUDA_R_32F);
        // Create dense vector Y
        cusparseCreateDnVec(&vecY, denseY.num_rows, denseY.entries, CUDA_R_32F);
    }

    void sparse_csr_matrix_dense_vector_multiplication_BUFFER_setting(cusparseHandle_t &handle, 
        cusparseSpMatDescr_t &matA, cusparseDnVecDescr_t &vecX, cusparseDnVecDescr_t &vecY, void* dBuffer, 
        cusparseOperation_t trans, float alpha = 1.0f) {
        //-------------------------------------------------------------------------
        size_t               bufferSize = 0;
        float                beta = 0.0f;
        //--------------------------------------------------------------------------
        // allocate an external buffer if needed
        cusparseSpMV_bufferSize(handle, trans,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);
    }

    // Deletion of SpMat, DnVec and dBuffer. 
    void sparse_csr_matrix_dense_vector_multiplication_free(cusparseSpMatDescr_t matA, 
        cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, void* dBuffer) {
        //--------------------------------------------------------------------------
        // destroy matrix/vector descriptors
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        //--------------------------------------------------------------------------
        // device memory deallocation
        cudaFree(dBuffer);
    }      
    
    // The multiplication of two dense vectors. 
    float dense_vectors_multiplication(Dense_vector<float> denseX, Dense_vector<float> denseY, float alpha = 1.0f) {
        cublasHandle_t     handle = NULL;
        cublasCreate(&handle);
        float* hResult = new float[1]();
        cublasSdot(handle, denseX.num_rows, denseX.entries, 1, denseY.entries, 1, hResult);
        float s = alpha * hResult[0];
        cublasDestroy(handle);
    
        delete[]hResult;
    
        return s;
    }

    // The addition of two dense vectors. 
    Dense_vector<float> dense_vector_addition(Dense_vector<float> denseX, Dense_vector<float> denseY, float alpha = 1.0f) {
        //--------------------------------------------------------------------------
        parallel_addition << <(denseY.num_rows + MAX_THREAD - 1) / MAX_THREAD, MAX_THREAD >> > (denseY.num_rows, alpha, denseX, denseY);
    
        return denseY;
    }

    // The summation of all the elements in a dense vector. 
    float dense_vector_summation(Dense_vector<float> denseX) {
        cublasHandle_t     handle = NULL;
        cublasCreate(&handle);
        float* hResult = (float*)malloc(sizeof(float));
        cublasSasum(handle, denseX.num_rows, denseX.entries, 1, hResult);
        cublasDestroy(handle);
    
        return hResult[0];
    }
    
    // The square of norm2 value of a dense vector. 
    float dense_vector_norm2_square(Dense_vector<float> denseX) {
        cublasHandle_t     handle = NULL;
        cublasCreate(&handle);
        float* hResult = (float*)malloc(sizeof(float));
        cublasSnrm2(handle, denseX.num_rows, denseX.entries, 1, hResult);
        cublasDestroy(handle);
    
        return hResult[0] * hResult[0];
    }
    
    // The norm2 value of a dense vector. 
    float dense_vector_norm2(Dense_vector<float> denseX) {
        cublasHandle_t     handle = NULL;
        cublasCreate(&handle);
        float* hResult = (float*)malloc(sizeof(float));
        cublasSnrm2(handle, denseX.num_rows, denseX.entries, 1, hResult);
        cublasDestroy(handle);
    
        return hResult[0];
    }

    // The conversion from CSR format to CSC format. 
    Sparse_CSR sparse_matrix_CSR_to_CSC(Sparse_CSR dM_CSR, cusparseHandle_t& handle) {
        int M_num_nonzeros = dM_CSR.num_nonzeros;
        int M_num_rows = dM_CSR.num_rows;
        int M_num_cols = dM_CSR.num_cols;
    
        int* dM_T_csrOffsets, * dM_T_columns;
        float* dM_T_values;
    
        cudaMalloc((void**)&dM_T_csrOffsets, (M_num_rows + 1) * sizeof(int));
        cudaMalloc((void**)&dM_T_columns, M_num_nonzeros * sizeof(int));
        cudaMalloc((void**)&dM_T_values, M_num_nonzeros * sizeof(float));

        size_t buffer_temp_size;
        cusparseCsr2cscEx2_bufferSize(handle, M_num_rows, M_num_cols, M_num_nonzeros, dM_CSR.values, dM_CSR.csrOffsets, dM_CSR.columns,
            dM_T_values, dM_T_csrOffsets, dM_T_columns, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffer_temp_size);
        void* buffer_temp = NULL;
        cudaMalloc(&buffer_temp, buffer_temp_size);
        cusparseCsr2cscEx2(handle, M_num_rows, M_num_cols, M_num_nonzeros, dM_CSR.values, dM_CSR.csrOffsets, dM_CSR.columns,
            dM_T_values, dM_T_csrOffsets, dM_T_columns, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer_temp);
    
        cudaFree(buffer_temp);
    
        Sparse_CSR dM_T = { M_num_rows, M_num_cols, M_num_nonzeros, dM_T_csrOffsets, dM_T_columns, dM_T_values };
    
        return dM_T;
    }

    // some information of gpu. 
    void process_info(int dev) {
        cout << setiosflags(ios::fixed) << setprecision(2);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, dev);
        cudaSetDevice(dev);
        cout << endl;
        cout << "GPU device " << dev << ": " << devProp.name << endl;
        cout << "Clock rate: " << devProp.clockRate * 1e-6f << " GHz" << endl;
        cout << "The number of SM: " << devProp.multiProcessorCount << endl;
        cout << "The number of shared memory for each block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
        cout << "Maximum number of threads in each block: " << devProp.maxThreadsPerBlock << endl;
        cout << "Maximum number of threads in each SM: " << devProp.maxThreadsPerMultiProcessor << endl;
        cout << "Maximum number of warps in each SM: " << devProp.maxThreadsPerMultiProcessor / 32 << endl;
        cout << endl;
        cout << setiosflags(ios::fixed) << setprecision(7);
    }

    SimRankStruct(string name, string file_label, float epsilon, float cvalue, int idx_base_type, string ldim_type, 
                int cuda_dev, int jump_times, bool dual_seeds) {
        process_info(cuda_dev);
        // cout.setf(ios::scientific);
        c = cvalue;
        sqrtC = sqrt(cvalue);
        eps = epsilon;
        delta = 0.01;
        jump_num = jump_times;
        dual_flag = dual_seeds;

        filelabel = file_label;
        hP_COO = inputGraph(name, idx_base_type, ldim_type);
        vert = hP_COO.num_rows;
        edges = hP_COO.num_nonzeros;
        walk_num = (long)((c * 0.00004) * (log(vert / delta) / log(2)) / (eps * eps));
        L = (int)((log(eps * (1 - c)) / log(c) - 1) + 1);

        cout << "eps: " << eps << endl;
        cout << setiosflags(ios::fixed) << setprecision(2);
        cout << "decay c: " << c << endl;
        cout << "Sampling times: " << walk_num << endl;
        cout << "Iteration times: " << L << endl;
        cout << setiosflags(ios::fixed) << setprecision(5);

        avg_time = 0;

        //--------------------------------------------------------------------------
        float t_allocation = 0;
        cudaEvent_t t_allocation_start, t_allocation_stop;
        float elapsedTime_allocation = 0.0;

        cudaEventCreate(&t_allocation_start);
        cudaEventCreate(&t_allocation_stop);
        cudaEventRecord(t_allocation_start, 0);
        //--------------------------------------------------------------------------
        P_num_nonzeros = hP_COO.num_nonzeros;
        P_num_rows = hP_COO.num_rows;
        P_num_cols = hP_COO.num_cols;
        vecS_num_rows = hP_COO.num_rows;

        cudaMalloc((void**)&dP_COO_rows, P_num_nonzeros * sizeof(int));
        cudaMalloc((void**)&dP_indegrees, P_num_rows * sizeof(int));
        cudaMalloc((void**)&dP_csrOffsets, (P_num_rows + 1) * sizeof(int));
        cudaMalloc((void**)&dP_columns, P_num_nonzeros * sizeof(int));
        cudaMalloc((void**)&dP_values, P_num_nonzeros * sizeof(float));

        cudaMemcpy(dP_indegrees, hP_COO.indegrees, P_num_rows * sizeof(int),
            cudaMemcpyHostToDevice);
        cudaMemcpy(dP_COO_rows, &hP_COO.rows[0], P_num_nonzeros * sizeof(int),
            cudaMemcpyHostToDevice);
        cudaMemcpy(dP_columns, &hP_COO.columns[0], P_num_nonzeros * sizeof(int),
            cudaMemcpyHostToDevice);
        //--------------------------------------------------------------------------
        cudaEventRecord(t_allocation_stop, 0);
        cudaEventSynchronize(t_allocation_stop);
        cudaEventElapsedTime(&elapsedTime_allocation, t_allocation_start, t_allocation_stop);
        t_allocation = (elapsedTime_allocation * 1000.0 / CLOCKS_PER_SEC);

        cout << "Allocate P takes " << t_allocation << " s." << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        float t_construct_P = 0;
        cudaEvent_t start4, stop4;
        float elapsedTime4 = 0.0;

        cudaEventCreate(&start4);
        cudaEventCreate(&stop4);
        cudaEventRecord(start4, 0);

        // Calculate the value array of matrix P (stored in CSR format) in parallel. 
        matrix_P_construction << < MAX_BLOCK, MAX_THREAD >> > (dP_indegrees, dP_columns, dP_values, P_num_nonzeros);

        cudaEventRecord(stop4, 0);
        cudaEventSynchronize(stop4);
        cudaEventElapsedTime(&elapsedTime4, start4, stop4);
        t_construct_P = (elapsedTime4 * 1000.0 / CLOCKS_PER_SEC);
        cout << "matrix P construction takes " << t_construct_P << " s." << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        cudaMalloc((void**)&dDiag_values, vecS_num_rows * sizeof(float));
        cudaMalloc((void**)&dVecMid_ent, vecS_num_rows * sizeof(float));
        dVecMid = { vecS_num_rows, dVecMid_ent };
        cudaMalloc((void**)&dTmpVec_ent, vecS_num_rows * sizeof(float));
        dTmpVecS = { vecS_num_rows, dTmpVec_ent };
        //--------------------------------------------------------------------------
    }

    ~SimRankStruct() {
        cudaFree(dP.csrOffsets);
        cudaFree(dP.columns);
        cudaFree(dP.values);
        cudaFree(dVecMid_ent);
        cudaFree(dTmpVec_ent);
        cudaFree(dDiag_values);
        vector<int>().swap(hP_COO.rows);
        vector<int>().swap(hP_COO.columns);
        cudaFree(dP_indegrees);
        delete[] hP_COO.indegrees;
    }

    void ClipSim(int u, string outputFile) {
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------    
        float t_coo2csr = 0;
        cudaEvent_t t_coo2csr_start, t_coo2csr_stop;
        float elapsedTime_coo2csr = 0.0;

        cudaEventCreate(&t_coo2csr_start);
        cudaEventCreate(&t_coo2csr_stop);
        cudaEventRecord(t_coo2csr_start, 0);
        //--------------------------------------------------------------------------
        cusparseHandle_t     handle = NULL;
        cusparseCreate(&handle);
        // Convert COO format to CSR format. (only the row array in COO needs to be compressed) 
        cusparseXcoo2csr(handle, dP_COO_rows, P_num_nonzeros, P_num_rows, dP_csrOffsets, CUSPARSE_INDEX_BASE_ZERO);
        //--------------------------------------------------------------------------
        cudaEventRecord(t_coo2csr_stop, 0);
        cudaEventSynchronize(t_coo2csr_stop);
        cudaEventElapsedTime(&elapsedTime_coo2csr, t_coo2csr_start, t_coo2csr_stop);
        t_coo2csr = (elapsedTime_coo2csr * 1000.0 / CLOCKS_PER_SEC);

        cout << "coo2csr takes " << t_coo2csr << " s." << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        dP = { P_num_rows, P_num_cols, P_num_nonzeros, dP_csrOffsets, dP_columns, dP_values };
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        float t_csr2csc = 0;
        cudaEvent_t t_csr2csc_start, t_csr2csc_stop;
        float elapsedTime_csr2csc = 0.0;

        cudaEventCreate(&t_csr2csc_start);
        cudaEventCreate(&t_csr2csc_stop);
        cudaEventRecord(t_csr2csc_start, 0);
        //--------------------------------------------------------------------------
        // CSR to CSC. Since we need to use P_T in the calculation of PPR. 
        dP_T = sparse_matrix_CSR_to_CSC(dP, handle);
        //--------------------------------------------------------------------------
        cudaEventRecord(t_csr2csc_stop, 0);
        cudaEventSynchronize(t_csr2csc_stop);
        cudaEventElapsedTime(&elapsedTime_csr2csc, t_csr2csc_start, t_csr2csc_stop);
        t_csr2csc = (elapsedTime_csr2csc * 1000.0 / CLOCKS_PER_SEC);

        cout << "csr2csc takes " << t_csr2csc << " s." << endl;
        //--------------------------------------------------------------------------
        cout << "======CSR and CSC conversion done!======\n" << endl;
        //--------------------------------------------------------------------------
        float t_total = 0;
        cudaEvent_t start_total, stop_total;
        float elapsedTime_total = 0.0;

        cudaEventCreate(&start_total);
        cudaEventCreate(&stop_total);
        cudaEventRecord(start_total, 0);
        //--------------------------------------------------------------------------
        int source = u;
        hVecS = new float[vecS_num_rows]();
        hVecS[source] = (1 - sqrtC) * 1.0f;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        float t_ppr = 0;
        cudaEvent_t t_ppr_start, t_ppr_stop;
        float elapsedTime_ppr = 0.0;

        cudaEventCreate(&t_ppr_start);
        cudaEventCreate(&t_ppr_stop);
        cudaEventRecord(t_ppr_start, 0);
        //--------------------------------------------------------------------------
        float* dPPR_ent;
        cudaMalloc((void**)&dPPR_ent, vecS_num_rows * sizeof(float));
        cudaMemcpy(dPPR_ent, hVecS, vecS_num_rows * sizeof(float),
                cudaMemcpyHostToDevice);
        Dense_vector<float> dPPR = { vecS_num_rows, dPPR_ent };
        //--------------------------------------------------------------------------
        dPPR_all = new Dense_vector<float>[L+1];
        //--------------------------------------------------------------------------
        // Allocate for PPR. 
        for (int i = 0; i < L + 1; i++) {
            float* dPPR_l_ent;
            cudaMalloc((void**)&dPPR_l_ent, vecS_num_rows * sizeof(float));
            if (i == 0)
                cudaMemcpy(dPPR_l_ent, hVecS, vecS_num_rows * sizeof(float),
                    cudaMemcpyHostToDevice);
            Dense_vector<float> dPPR_l = { vecS_num_rows, dPPR_l_ent };
            dPPR_all[i] = dPPR_l;
        }
        //--------------------------------------------------------------------------
        // Calculate PPR. 
        for (int ell = 1; ell < L + 1; ell++) {
            sparse_csr_matrix_dense_vector_multiplication(handle, dP, dPPR_all[ell - 1], dPPR_all[ell],
                CUSPARSE_OPERATION_NON_TRANSPOSE, sqrtC);
            dPPR = dense_vector_addition(dPPR_all[ell], dPPR);
        }
        // cusparseSpMatDescr_t sparseMatP;
        // cusparseDnVecDescr_t denseVecX, denseVecY;
        // void* dBuffers = NULL;
        // //--------------------------------------------------------------------------
        // float beta = 0.0f;
        // sparse_csr_matrix_dense_vector_multiplication_MAT_setting(handle, dP, sparseMatP); // mat
        // for (int ell = 1; ell < L + 1; ell++) {
        //     sparse_csr_matrix_dense_vector_multiplication_VEC_setting(handle, dPPR_all[ell - 1], dPPR_all[ell], 
        //         denseVecX, denseVecY); // vec
        //     if (ell == 1)
        //         sparse_csr_matrix_dense_vector_multiplication_BUFFER_setting(handle, sparseMatP, denseVecX, denseVecY, 
        //             dBuffers, CUSPARSE_OPERATION_NON_TRANSPOSE, sqrtC); // buffer
        //     // execute SpMV
        //     cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &sqrtC, sparseMatP, denseVecX, &beta, denseVecY, CUDA_R_32F,
        //         CUSPARSE_MV_ALG_DEFAULT, dBuffers);
        //     dPPR = dense_vector_addition(dPPR_all[ell], dPPR);
        // }
        // sparse_csr_matrix_dense_vector_multiplication_free(sparseMatP, denseVecX, denseVecY, dBuffers) ;
        //--------------------------------------------------------------------------
        cudaEventRecord(t_ppr_stop, 0);
        cudaEventSynchronize(t_ppr_stop);
        cudaEventElapsedTime(&elapsedTime_ppr, t_ppr_start, t_ppr_stop);
        t_ppr = (elapsedTime_ppr * 1000.0 / CLOCKS_PER_SEC);

        cout << "Personalized PageRank takes " << t_ppr << " s." << endl;
        cout << "======Personalized PageRank done!======\n" << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        float t_D = 0;
        cudaEvent_t t_D_start, t_D_stop;
        float elapsedTime_D = 0.0;

        cudaEventCreate(&t_D_start);
        cudaEventCreate(&t_D_stop);
        cudaEventRecord(t_D_start, 0);
        //--------------------------------------------------------------------------
        float t_allocate_idx = 0;
        cudaEvent_t t_allocate_idx_start, t_allocate_idx_stop;
        float elapsedTime_allocate_idx = 0.0;

        cudaEventCreate(&t_allocate_idx_start);
        cudaEventCreate(&t_allocate_idx_stop);
        cudaEventRecord(t_allocate_idx_start, 0);
        //--------------------------------------------------------------------------
        int* dEnd_vert_idx_ent;
        cudaMalloc((void**)&dEnd_vert_idx_ent, TOTAL_WARPS * sizeof(int));
        Dense_vector<int> dEnd_vert_idx = { TOTAL_WARPS, dEnd_vert_idx_ent };
        unsigned int* dMeets_count;
        cudaMalloc((void**)&dMeets_count, vecS_num_rows * sizeof(unsigned int));
        int* dPortion_walk;
        cudaMalloc((void**)&dPortion_walk, vecS_num_rows * sizeof(int));
        int* dPortion_walk_dup;
        cudaMalloc((void**)&dPortion_walk_dup, vecS_num_rows * sizeof(int));
        int* dEach_portion_walk_num;
        cudaMalloc((void**)&dEach_portion_walk_num, vecS_num_rows * sizeof(int));

        Dense_vector<int> dIndegree = { vecS_num_rows, dP_indegrees };
        //--------------------------------------------------------------------------
        cudaEventRecord(t_allocate_idx_stop, 0);
        cudaEventSynchronize(t_allocate_idx_stop);
        cudaEventElapsedTime(&elapsedTime_allocate_idx, t_allocate_idx_start, t_allocate_idx_stop);
        t_allocate_idx = (elapsedTime_allocate_idx * 1000.0 / CLOCKS_PER_SEC);

        cout << "Allocate several arrays takes " << t_allocate_idx << " s." << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        float t_para_RW = 0;
        cudaEvent_t start, stop;
        float elapsedTime = 0.0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Usually, the number of vertices is larger than the number of threads. 
        // Thus, there are more than one vertices corresponding to a thread. 
        // e.g., there are 3 threads and 8 vertices. 
        // Thread 0 for vertex 0, 3, 6; thread 1 for vertex 1, 4, 7; thread 2 for vertex 2, 5. 
        // Then, "dEnd_vert_idx" is [6, 7, 5]. 
        find_end_vert_per_thread << < MAX_BLOCK, MAX_THREAD >> > (dEnd_vert_idx, vecS_num_rows);

        // There are a number of random walks starting from each vertex. 
        // The number of random walks for each vertex varies according to PPR. 
        // We determine the number of portions of random walks for each vertex. 
        // Each portion contains fixed number of random walks. This can reduce the number of atomic operations. 
        // decide_each_vert_portion_walk_num << < MAX_BLOCK, MAX_THREAD >> > (dPortion_walk, dEach_portion_walk_num, dPPR, walk_num, vector_norm2);
        decide_each_vert_portion_walk_num << < MAX_BLOCK, MAX_THREAD >> > (dPortion_walk, dEach_portion_walk_num, dPPR, walk_num);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        cudaMemcpy(dPortion_walk_dup, dPortion_walk, vecS_num_rows * sizeof(int), cudaMemcpyDeviceToDevice);

        // Operate random walks in parallel. 
        // Obtain the number of pairs of random walks meeting at a vertex. 
        parallel_random_walk_v3_2 << < MAX_BLOCK, MAX_THREAD >> > (rand(), dP_T, dEnd_vert_idx, 
                dPortion_walk, dEach_portion_walk_num, c, dMeets_count, jump_num, dual_flag);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        t_para_RW = (elapsedTime * 1000.0 / CLOCKS_PER_SEC);

        cout << "RW takes " << t_para_RW << " s." << endl;
        //--------------------------------------------------------------------------
        float t_para_D = 0;
        cudaEvent_t start2, stop2;
        float elapsedTime2 = 0.0;

        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);
        cudaEventRecord(start2, 0);

        // Calculate matrix D according to "dMeets_count". 
        calculate_matrix_D << < MAX_BLOCK, MAX_THREAD >> > (dDiag_values, c, dIndegree, dMeets_count, dPortion_walk_dup, dEach_portion_walk_num);

        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        cudaEventElapsedTime(&elapsedTime2, start2, stop2);
        t_para_D = (elapsedTime2 * 1000.0 / CLOCKS_PER_SEC);

        cout << "calculate matrix D takes " << t_para_D << " s." << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        float t_free_idx = 0;
        cudaEvent_t t_free_idx_start, t_free_idx_stop;
        float elapsedTime_free_idx = 0.0;

        cudaEventCreate(&t_free_idx_start);
        cudaEventCreate(&t_free_idx_stop);
        cudaEventRecord(t_free_idx_start, 0);
        //--------------------------------------------------------------------------
        cudaFree(dEnd_vert_idx.entries);
        cudaFree(dMeets_count);
        cudaFree(dPortion_walk);
        cudaFree(dPortion_walk_dup);
        cudaFree(dEach_portion_walk_num);
        //--------------------------------------------------------------------------
        cudaEventRecord(t_free_idx_stop, 0);
        cudaEventSynchronize(t_free_idx_stop);
        cudaEventElapsedTime(&elapsedTime_free_idx, t_free_idx_start, t_free_idx_stop);
        t_free_idx = (elapsedTime_free_idx * 1000.0 / CLOCKS_PER_SEC);

        cout << "free arrays takes " << t_free_idx << " s." << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        cudaEventRecord(t_D_stop, 0);
        cudaEventSynchronize(t_D_stop);
        cudaEventElapsedTime(&elapsedTime_D, t_D_start, t_D_stop);
        t_D = (elapsedTime_D * 1000.0 / CLOCKS_PER_SEC);

        cout << endl;
        cout << "Sampling takes " << t_D << " s." << endl;
        cout << "======Sampling matrix D done!======\n" << endl;
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        float t_SpMxV = 0;
        cudaEvent_t t_SpMxV_start, t_SpMxV_stop;
        float elapsedTime_SpMxV = 0.0;

        cudaEventCreate(&t_SpMxV_start);
        cudaEventCreate(&t_SpMxV_stop);
        cudaEventRecord(t_SpMxV_start, 0);
        //--------------------------------------------------------------------------
        cudaMalloc((void**)&dVecS_ent, vecS_num_rows * sizeof(float));
        Dense_vector<float> dVecS = { vecS_num_rows, dVecS_ent };
        float transSqrtC = 1.0 / (1.0 - sqrtC);
        // Calculate the SimRank vector with SpMxV. 
        diag_matrix_dense_vector_multiplication << < (vecS_num_rows + 1023) / 1024, 1024 >> > \
            (dDiag_values, dPPR_all[L], dVecS, transSqrtC);
        //--------------------------------------------------------------------------
        for (int ell = 1; ell < L + 1; ell++) {
            sparse_csr_matrix_dense_vector_multiplication(handle, dP_T, dVecS, dTmpVecS,
                CUSPARSE_OPERATION_NON_TRANSPOSE, sqrtC);
            diag_matrix_dense_vector_multiplication << < (vecS_num_rows + 1023) / 1024, 1024 >> > \
                (dDiag_values, dPPR_all[L - ell], dVecMid, transSqrtC);
            dVecS = dense_vector_addition(dTmpVecS, dVecMid);
        }
        // sparse_csr_matrix_dense_vector_multiplication_setting(handle, dP_T, dVecS, dTmpVecS, 
        //     sparseMatP, denseVecX, denseVecY, dBuffers, CUSPARSE_OPERATION_NON_TRANSPOSE, sqrtC);
        // //--------------------------------------------------------------------------
        // for (int ell = 1; ell < L + 1; ell++) {
        //     // execute SpMV
        //     cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &sqrtC, sparseMatP, denseVecX, &beta, denseVecY, 
        //         CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dBuffers);
        //     diag_matrix_dense_vector_multiplication << < (vecS_num_rows + 1023) / 1024, 1024 >> > \
        //         (dDiag_values, dPPR_all[L - ell], dVecMid, transSqrtC);
        //     dVecS = dense_vector_addition(dTmpVecS, dVecMid);
        // }
        // sparse_csr_matrix_dense_vector_multiplication_free(sparseMatP, denseVecX, denseVecY, dBuffers) ;
        cusparseDestroy(handle);
        clock_t t2 = clock();
        //--------------------------------------------------------------------------
        cudaEventRecord(t_SpMxV_stop, 0);
        cudaEventSynchronize(t_SpMxV_stop);
        cudaEventElapsedTime(&elapsedTime_SpMxV, t_SpMxV_start, t_SpMxV_stop);
        t_SpMxV = (elapsedTime_SpMxV * 1000.0 / CLOCKS_PER_SEC);

        cout << "SpMxV takes " << t_SpMxV << " s." << endl;
        cout << "======SimRank (SpMxV) done!======\n" << endl;
        //--------------------------------------------------------------------------
        cudaEventRecord(stop_total, 0);
        cudaEventSynchronize(stop_total);
        cudaEventElapsedTime(&elapsedTime_total, start_total, stop_total);
        t_total = (elapsedTime_total * 1000.0 / CLOCKS_PER_SEC);
        //--------------------------------------------------------------------------
        //----------------------------------------------------------------------------
        cout << "======SimRank done!======\n" << endl;

        cout << "============SimRank query CudaEvent takes " << t_total << " s. ============" << endl;
        avg_time += t_total;
        //--------------------------------------------------------------------------
        // device result check
        cudaMemcpy(hVecS, dVecS.entries, dVecS.num_rows * sizeof(float), cudaMemcpyDeviceToHost);
        hVecS[source] = 1.0f;

        cudaFree(dP_T.csrOffsets);
        cudaFree(dP_T.columns);
        cudaFree(dP_T.values);

        for (int i = 0; i < L + 1; i++) {
            cudaFree(dPPR_all[i].entries);
        }
        cudaFree(dVecS_ent);
        cudaFree(dPPR_ent);

        ofstream fout;
        fout.open(outputFile);
        fout.setf(ios::fixed, ios::floatfield);
        fout.precision(15);
        if (!fout) cout << "Fail to open the file" << endl;
        for (int j = 0; j < dVecS.num_rows; j++) {
            if (hVecS[j] > 0.0) {
                fout << j << " " << hVecS[j] << endl;
            }
        }
        fout.close();

        delete[] hVecS;
    }
};