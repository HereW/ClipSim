## Tested Environment:
- Ubuntu 18.04.5
- C++ 17
- GCC 5.4.0


## Compile the code:
```
make
```


## Parameters:
- -d \<dataset\> 
- -f \<filelabel\>
- [-algo \<algorithm\> (options: GEN_QUERY, ClipSim) (default ClipSim)]
- [-e \<epsilon\> (default 0.001)]
- [-qn \<querynum\> (default 50)]
- [-c \<damping factor\> (default 0.8)]
- [-idx \<index base type\> (options: 0, 1) (default 0)]
- [-ld \<leading dim\> (options: row, col) (default row)]
- [-cuda \<cuda device\> (default 0)]
- [-jump \<jump number\> (default 20)]
- [-dual \<dual seeds\> (default true)]


## Run the example:
(1) Generate query nodes:   
```
./ClipSim -d dataset/toy_graph.txt -f toy_graph -algo GEN_QUERY -qn 20 -idx 0
```
(2) Run SimRank algorithm: 
```
./ClipSim -d dataset/toy_graph.txt -f toy_graph -algo ClipSim -e 1.0e-06 -qn 1 -c 0.6 -idx 0 -ld row
```


## Instructions:
(1) datatset/: datasets files.  
(2) query/: query nodes files.  
(3) results/: SimRank results files (only output nonzero SimRank values).  