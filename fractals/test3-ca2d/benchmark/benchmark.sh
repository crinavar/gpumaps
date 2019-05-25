#!/bin/bash
for b in `seq 0 5`;
do
    for rl in `seq 0 16`;
    do
        echo "**** mapping Sirpinski rlevel=${rl} bpower=${b}"
        ../prog ${b} ${rl} "mapstat_b${b}.dat"
        echo " "
    done
done 
