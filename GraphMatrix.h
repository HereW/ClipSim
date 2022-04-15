#pragma once
#ifndef GRAPHMATRIX_H
#define GRAPHMATRIX_H

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <unordered_map>
using namespace std;

struct Sparse_COO
{
    int  num_rows, num_cols, num_nonzeros;
    vector<int>     rows;
    vector<int>     columns;
    // vector<float>   values;
    // unordered_map<int, int> indegrees;
    int  *          indegrees;
    // int  *          outdegrees;
};

//struct Sparse_dCOO
//{
//    int  num_rows, num_cols, num_nonzeros;
//    int     * rows, * columns;
//    float   * values;
//};

struct Sparse_CSR
{
    long long    num_rows, num_cols, num_nonzeros;
    int          * csrOffsets, * columns;
    float        * values;
};

template<class T>
struct Dense_matrix
{
    long    num_rows, num_cols;
    T       * entries;
};

template<class T>
struct Dense_vector
{
    long    num_rows;
    T       * entries;
};

#endif