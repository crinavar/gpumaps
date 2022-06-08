cd test1-map
./benchmark-blockconf.sh 0    0 5 1    1024 78848 4096  10 25  ./bin/test1-map    TITAN-RTX    sm_75
cd ..

cd test2-accum
./benchmark-blockconf.sh 0    0 5 1    1024 78848 4096  10 25  ./bin/test2-accum    TITAN-RTX    sm_75
cd ..

cd test3-edm2d
./benchmark-blockconf.sh 0    0 5 1    1024 78848 4096  10 25  ./bin/test3-edm2d    TITAN-RTX    sm_75
cd ..

cd test4-ca2d
./benchmark-blockconf.sh 0    0 5 1    1024 78848 4096  10 25  ./bin/test4-ca2d    TITAN-RTX    sm_75
cd ..


cd test1-map
./benchmark-blockconf.sh 1    0 5 1    1024 78848 4096  10 25  ./bin/test1-map    TITAN-V    sm_70
cd ..

cd test2-accum
./benchmark-blockconf.sh 1    0 5 1    1024 78848 4096  10 25  ./bin/test2-accum    TITAN-V    sm_70
cd ..

cd test3-edm2d
./benchmark-blockconf.sh 1    0 5 1    1024 78848 4096  10 25  ./bin/test3-edm2d    TITAN-V    sm_70
cd ..

cd test4-ca2d
./benchmark-blockconf.sh 1    0 5 1    1024 78848 4096  10 25  ./bin/test4-ca2d    TITAN-V    sm_70
cd ..
