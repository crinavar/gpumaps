#cd test1-map
#./benchmark-blockconf.sh 0    4 5 1    1024 78848 2048  10 10  ./bin/test1-map    TITAN-RTX    sm_75
#cd ..
#
#cd test2-accum
#./benchmark-blockconf.sh 0    4 5 1    1024 78848 2048  10 10  ./bin/test2-accum    TITAN-RTX    sm_75
#cd ..
#
#cd test3-edm2d
#./benchmark-blockconf.sh 0    4 5 1    1024 78848 2048  10 10  ./bin/test3-edm2d    TITAN-RTX    sm_75
#cd ..
#
#cd test4-ca2d
#./benchmark-blockconf.sh 0    5 5 1    1024 78848 2048  10 10  ./bin/test4-ca2d    TITAN-RTX    sm_75
#cd ..


cd test1-map
./benchmark-blockconf.sh 1    4 5 1    1024 50176 2048  20 3  ./bin/test1-map    TITAN-V    sm_70
cd ..

cd test2-accum
./benchmark-blockconf.sh 1    4 5 1    1024 50176 2048  20 5 ./bin/test2-accum    TITAN-V    sm_70
cd ..

cd test3-edm2d
./benchmark-blockconf.sh 1    4 5 1    1024 50176 2048  20 5 ./bin/test3-edm2d    TITAN-V    sm_70
cd ..

cd test4-ca2d
./benchmark-blockconf.sh 1    4 5 1    1024 37888 2048  20 5 ./bin/test4-ca2d    TITAN-V    sm_70
cd ..
