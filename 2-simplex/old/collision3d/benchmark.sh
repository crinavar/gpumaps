#!/bin/bash
DELTA=1024
for i in `seq 1 30`;
do
            c=$(($i * $DELTA))
            echo "**** processing size ${c} with mode td${1}"
            ./extd ${c} 1
            ./extd ${c} 2
            ./extd ${c} 3
			./extd ${c} 4
            ./extd ${c} 5 64
            echo " "
done 
