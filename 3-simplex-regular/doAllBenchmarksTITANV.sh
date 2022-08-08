#cd test1-map
#./benchmark-blockconf.sh  1 "TITAN-V"  "-arch=sm_70"    0 3 1    1 10 1     30
#cd ..
#
#cd test2-accumulate
#./benchmark-blockconf.sh  1 "TITAN-V"  "-arch=sm_70"    0 3 1    1 10 1     30
#cd ..

cd test3-ca2d
./benchmark-blockconf.sh  1 "TITAN-V"  "-arch=sm_70"    0 3 1    1 10 1     30
cd ..
