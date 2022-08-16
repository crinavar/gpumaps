cd test1-map
./benchmark-blockconf.sh  0 "A100"  "-arch=sm_80"    1 3 1    1 11 1     20
cd ..

cd test2-accumulate
./benchmark-blockconf.sh  0 "A100"  "-arch=sm_80"    1 3 1    1 11 1     20
cd ..

cd test3-ca2d
./benchmark-blockconf.sh  0 "A100"  "-arch=sm_80"    1 3 1    1 10 1     20
cd ..
