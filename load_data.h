#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <time.h>
#include "GraphMatrix.h"

using namespace std;

Sparse_COO inputGraph(string filename, int idx_base, string ldim) {
    long m = 0;
    long n = 0;

    clock_t t1 = clock();

    ifstream infile(filename.c_str());

    // obtain n, m. 
    int f;
    int t;
    while (infile >> f>> t) {
        m++;
        n = (n > f ? n : f);
        n = (n > t ? n : t);
    }
    n += (1 - idx_base);

    infile.clear();
    infile.seekg(0);

    infile.close();

    //unordered_map<int, int> indegree;

    Sparse_COO M_COO;

    // read graph and get degree info
    
    // n = 2393285;
    // m = 128194008;

    vector<int> M_COO_rows(m);
    vector<int> M_COO_cols(m);
    
    int * indegree_arr = new int[n]();
    
    long i = 0;

    ifstream infile2(filename.c_str());
    
    int from;
    int to;
    while (infile2 >> from >> to) {
        if (ldim == "row") {
            indegree_arr[to - idx_base]++;
            M_COO_rows[i] = from - idx_base;
            M_COO_cols[i] = to - idx_base;
            i++;
        }
        else {
            indegree_arr[from - idx_base]++;
            M_COO_rows[i] = to - idx_base;
            M_COO_cols[i] = from - idx_base;
            i++;
        }
    }

    infile2.clear();
    infile2.seekg(0);

    infile2.close();

    M_COO.num_rows = n;
    M_COO.num_cols = n;
    M_COO.num_nonzeros = m;
    M_COO.indegrees = indegree_arr;
    M_COO.rows = M_COO_rows;
    M_COO.columns = M_COO_cols;

    clock_t t2 = clock();
    cout << endl;
    cout << "Nodes n=" << n << endl;
    cout << "Edges m=" << m << endl;
    cout << endl;
    cout << "reading in graph takes " << (t2 - t1) / (1.0 * CLOCKS_PER_SEC) << " s." << endl;
    cout << "====Graphs reading done!====\n" << endl;

    return M_COO;
}