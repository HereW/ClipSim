ClipSim:main.cu SimRankStruct.h SimRankKernels.h load_data.h GraphMatrix.h
	nvcc -std=c++17 -o $@ $< -lcublas -lcusparse
.PHONY:clean
clean:
	-rm *.o