#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <iostream>
#include <string.h>
#include <sstream>
#include <cstdlib>
#include <vector>
#include "SimRankStruct.h"
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>


int mkpath(string s, mode_t mode = 0755) {
    size_t pre = 0, pos;
    string dir;
    int mdret;
    if (s[s.size() - 1] != '/') {
        s += '/';
    }
    while ((pos = s.find_first_of('/', pre)) != string::npos) {
        dir = s.substr(0, pos++);
        pre = pos;
        if (dir.size() == 0) continue;
        if ((mdret = ::mkdir(dir.c_str(), mode)) && errno != EEXIST) {
            return mdret;
        }
    }
    return mdret;
}



void usage() {
    cout << "./testSimRank -d <dataset> -f <filelabel> -algo <Algorithm> [-e epsilon (default 0.001)] [-qn querynum (default 50)]" << endl;
}


int check_inc(int i, int max) {
    if (i == max) {
        usage();
        exit(1);
    }
    return i + 1;
}


int main(int argc, char** argv) {
    int i = 1;
    char* endptr;
    string filename;
    string outputname;
    string filelabel;
    string queryname;
    int querynum = -1;
    float eps = 0.001;
    float cvalue = 0.8;
    int idx_base = 0;
    string ldim;
    int dev = 0;
    int jump_num = 5;
    bool dual_flag = true;
    string algo = "ClipSim";
    if (argc < 1) {
        usage();
        exit(1);
    }
    while (i < argc) {
        if (!strcmp(argv[i], "-d")) {
            i = check_inc(i, argc);
            filename = argv[i];
        }
        else if (!strcmp(argv[i], "-f")) {
            i = check_inc(i, argc);
            filelabel = argv[i];
        }
        else if (!strcmp(argv[i], "-algo")) {
            i = check_inc(i, argc);
            algo = argv[i];
        }
        else if (!strcmp(argv[i], "-e")) {
            i = check_inc(i, argc);
            eps = strtod(argv[i], &endptr);
            if ((eps == 0 || eps > 1) && endptr) {
                cerr << "Invalid eps argument" << endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "-c")) {
            i = check_inc(i, argc);
            cvalue = strtod(argv[i], &endptr);
            if ((cvalue == 0 || cvalue > 1) && endptr) {
                cerr << "Invalid c argument" << endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "-qn")) {
            i = check_inc(i, argc);
            querynum = strtod(argv[i], &endptr);
            if ((querynum < 0) && endptr) {
                cerr << "Invalid query number argument" << endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "-o")) {
            i = check_inc(i, argc);
            outputname = argv[i];
        }
        else if (!strcmp(argv[i], "-idx")) {
            i = check_inc(i, argc);
            idx_base = strtod(argv[i], &endptr);
            if ((idx_base != 1) && (idx_base != 0) && endptr) {
                cerr << "Invalid index base type argument" << endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "-ld")) {
            i = check_inc(i, argc);
            ldim = argv[i];
            // if ((ldim != "row") && (idx_base != "col") && endptr) {
            //     cerr << "Invalid leading dimension type argument" << endl;
            //     exit(1);
            // }
        }
        else if (!strcmp(argv[i], "-cuda")) {
            i = check_inc(i, argc);
            dev = strtod(argv[i], &endptr);
        }
        else if (!strcmp(argv[i], "-jump")) {
            i = check_inc(i, argc);
            jump_num = strtod(argv[i], &endptr);
        }
        else if (!strcmp(argv[i], "-dual")) {
            i = check_inc(i, argc);
            istringstream(argv[i]) >> std::boolalpha >> dual_flag;
        }
        else {
            usage();
            exit(1);
        }
        i++;
    }

    SimRankStruct sim = SimRankStruct(filename, filelabel, eps, cvalue, idx_base, ldim, dev, jump_num, dual_flag);
    cout << "dual_flag: " << dual_flag << endl;

    if (querynum == -1) {
        querynum = max((int)sim.vert, 50);
    }


    // Generate query nodes
    //
    if (algo == "GEN_QUERY") {
        stringstream ss_gen;
        ss_gen << "query/";
        mkpath(ss_gen.str());
        ofstream data_idx("query/" + filelabel + ".query");
        for (int i = 0; i < querynum; i++) {
            data_idx << rand() % sim.vert << "\n";
        }
        data_idx.close();
        return EXIT_SUCCESS;
    }


    // ClipSim Algorithm
    //
    if (algo == "ClipSim") {
        queryname = "query/" + filelabel + ".query";
        ifstream query(queryname);
        cout << "querynum=" << querynum << endl;
        for (int i = 0; i < querynum; i++) {
            int nodeId;
            query >> nodeId;
            cout << i << ": " << nodeId << endl;
            stringstream ss, ss_dir;
            ss_dir << "results/" << filelabel << "/" << eps << "/";
            mkpath(ss_dir.str());
            ss << ss_dir.str() << nodeId << ".txt";
            outputname = ss.str();
            sim.ClipSim(nodeId, outputname);
        }
        query.close();

        cout << "\n====Query done!====\n" << endl;
        float avg_querytime = sim.avg_time / (float)querynum;
        cout << "Average Query Time: " << avg_querytime << " s\n" << endl;
    }

    return 0;
}