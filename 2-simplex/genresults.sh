#!/bin/bash

# do the experimentation
echo "starting experiment"
cd edm1d
sh benchmark.sh
cd ../edm2d
sh benchmark.sh
cd ../edm3d
sh benchmark.sh
cd ../edm4d
sh benchmark.sh
cd ../dummy
sh benchmark.sh
cd ../collision1d
sh benchmark.sh
cd ../collision2d
sh benchmark.sh
cd ../collision3d
sh benchmark.sh
cd ../

# join the results for later plotting
./benchmark_joiner edm1d/
./benchmark_joiner edm2d/
./benchmark_joiner edm3d/
./benchmark_joiner edm4d/
./benchmark_joiner dummy/
./benchmark_joiner collision1d/
./benchmark_joiner collision2d/
./benchmark_joiner collision3d/
echo "done."
