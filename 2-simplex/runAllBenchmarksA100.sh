#cd test1-map
#./benchmark-blockconf.sh 0    3 5 1    1024 99328 2048  10 25  ./bin/test1-map    A100    sm_80
#cd ..
#
#cd test2-accum
#./benchmark-blockconf.sh 0    3 5 1    1024 99328 2048  10 25  ./bin/test2-accum    A100    sm_80
#cd ..
#
#cd test3-edm2d
#./benchmark-blockconf.sh 0    5 5 1    1024 99328 2048  6 10  ./bin/test3-edm2d    A100    sm_80
#cd ..

cd test4-ca2d
./benchmark-blockconf.sh 0    4 5 1    1024 62464 2048  10 2  ./bin/test4-ca2d    A100    sm_80
cd ..
