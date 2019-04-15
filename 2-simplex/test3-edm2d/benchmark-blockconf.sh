#!/bin/bash
if [ "$#" -ne 11 ]; then
    echo "run as ./benchmark-blockconf.sh    DEV     STARTB ENDB DB    STARTN ENDN DN     KREPEATS SAMPLES BINARY OUTFILE"
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
BINARY=${10}
OUTFILE=${11}
HFACTOR=20
METHODS=("BBox" "Lambda" "Rectangle" "Trapezoid")
NM=$((${#METHODS[@]}))
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0
# show time
echo "STARTING TIME"
timedatectl
for B in `seq ${STARTB} ${DB} ${ENDB}`;
do
    echo "Benchmarking for B=${B} METHODS=${NM}"
    echo "Compiling with BSIZE3D=$B"
    LB=$((${B} * ${B}))
    COMPILE=`make BSIZE1D=${LB} BSIZE2D=${B} HADO_FACTOR=${HFACTOR}`
    echo ${COMPILE}
    for N in `seq ${STARTN} ${DN} ${ENDN}`;
    do
        echo "DEV=${DEV}  N=${N} B=${B} R=${R}"
        echo -n "${N}   ${B}    " >> data/${OUTFILE}_B${B}.dat
        for q in `seq 1 ${NM}`;
        do
            M=0
            S=0
            # Chosen MAP
            echo "./${BINARY} ${DEV}    ${N} ${R}    ${q}"
            echo -n "[WARMUP] ${METHODS[$(($q-1))]} ($q) map (${SAMPLES} Samples)................"
            for k in `seq 1 ${SAMPLES}`;
            do
                x=`./${BINARY} ${DEV} ${N} ${R} ${q}`
            done
            echo "done"
            echo -n "[BENCHMARK] ${METHODS[$(($q-1))]} ($q) map (${SAMPLES} Samples)............."
            for k in `seq 1 ${SAMPLES}`;
            do
                x=`./${BINARY} ${DEV} ${N} ${R} ${q}`
                oldM=$M;
                M=$(echo "scale=10;  $M+($x-$M)/$k"           | bc)
                S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | bc)
            done
            echo "done"
            MEAN=$M
            VAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"  | bc)
            STDEV=$(echo "scale=10; sqrt(${VAR})"       | bc)
            STERR=$(echo "scale=10; ${STDEV}/sqrt(${SAMPLES})" | bc)
            TMEAN[$q]=${MEAN}
            TVAR[$q]=${VAR}
            TSTDEV[$q]=${STDEV}
            TSTERR[$q]=${STERR}
            echo "---> B=${B} N=${N} --> (MEAN, VAR, STDEV, STERR) -> (${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]})"
            echo -n "${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]}         " >> data/${OUTFILE}_B${B}.dat
            echo " "
        done
        echo " " >> data/${OUTFILE}_B${B}.dat
        echo ""
        echo ""
        echo ""
    done 
    echo " "
done 
echo "END TIME"
timedatectl
echo "*******************************"
