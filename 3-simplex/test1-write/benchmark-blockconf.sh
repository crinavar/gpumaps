#!/bin/sh
if [ "$#" -ne 9 ]; then
    echo "run as ./benchmark-blockconf.sh DEV STARTB ENDB DB STARTN ENDN DN KREPEATS SAMPLES"
    exit;
fi
DEV=$1
STARTB=$2
ENDB=$3
DB=$4
STARTN=$5
ENDN=$6
DN=$7
R=$8
SAMPLES=$9
for B in `seq ${STARTB} ${DB} ${ENDB}`;
do
    echo "Benchmarking for B=${B}"
    for N in `seq ${STARTN} ${DN} ${ENDN}`;
    do
        echo "B=${B}            DEV=${DEV}  N=${N} R=${R}"

        echo -n "BB  map (${SAMPLES} Samples)............."
        M=0
        S=0
        for k in `seq 1 ${SAMPLES}`;
        do
            x=`./test1-write ${DEV} ${N} ${B} ${R} 0`
            oldM=$M;
            M=$(echo "scale=10;  $M+($x-$M)/$k"           | bc)
            S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | bc)
        done
        BBMEAN=$M
        BBVAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"  | bc)
        BBSTDEV=$(echo "scale=10; sqrt(${BBVAR})"       | bc)
        BBSTERR=$(echo "scale=10; ${BBSTDEV}/sqrt(${SAMPLES})" | bc)
        echo "done"


        echo -n "LAM map (${SAMPLES} Samples)............."
        M=0;
        S=0;
        for k in `seq 1 ${SAMPLES}`;
        do
            x=`./test1-write ${DEV} ${N} ${B} ${R} 1`
            oldM=$M;
            M=$(echo "scale=10;  $M+($x-$M)/$k"           | bc)
            S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | bc)
        done
        LAMMEAN=$M
        LAMVAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"   | bc)
        LAMSTDEV=$(echo "scale=10; sqrt(${LAMVAR})"      | bc)
        LAMSTERR=$(echo "scale=10; ${LAMSTDEV}/sqrt(${SAMPLES})" | bc)
        echo "done"

        echo "B=${B} N=${N} --> BB ${BBMEAN} [ms] (${BBVAR}, ${BBSTDEV}, ${BBSTERR}) LAM ${LAMMEAN} [ms] (${LAMVAR}, ${LAMSTDEV}, ${LAMSTERR}"
        echo "${N}   ${B}    ${BBMEAN}    ${BBVAR}    ${BBSTDEV}   ${BBSTERR}        ${LAMMEAN} ${LAMVAR} ${LAMSTDEV} ${LAMSTERR}" >> data/blockconf_B${B}.dat
        echo " "
    done 
    echo " "
done 
