cd test1-map
./benchmark-blockconf.sh  0 "TITAN-RTX"  "-arch=sm_75"    0 3 1    1 10 1     20
cd ..

cd test2-accumulate
./benchmark-blockconf.sh  0 "TITAN-RTX"  "-arch=sm_75"    0 3 1    1 10 1     20
cd ..

cd test3-ca2d
./benchmark-blockconf.sh  0 "TITAN-RTX"  "-arch=sm_75"    0 3 1    1 10 1     20
cd ..
