#!/bin/bash
DELTA=1024
for i in `seq 1 30`;
do
            c=$(($i * $DELTA))
            echo "**** processing size ${c} with mode td${1}"
            ./extd ${c} 1
            ./extd ${c} 6
            ./extd ${c} 7
            ./extd ${c} 8 64
            ./extd ${c} 2
            echo " "
done 
