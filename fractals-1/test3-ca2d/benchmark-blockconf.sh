#!/bin/bash
if [ "$#" -ne 9 ]; then
    echo "run as ./benchmark-blockconf.sh    DEV     STARTB ENDB DB    STARTR ENDR DR     KREPEATS BINARY"
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
BINARY=${9}
OUTFILE="d"
METHODS=("BBox" "Compressed" "Lambda")
NM=$((${#METHODS[@]}))
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0
# show time
echo "STARTING TIME"
timedatectl
for BP in `seq ${STARTB} ${DB} ${ENDB}`;
do
    echo "Benchmarking for BPOWER=${BP} METHODS=${NM}"
    BS=$((2**${BP}))
    LB=$((${BS} * ${BS}))
    for N in `seq ${STARTN} ${DN} ${ENDN}`;
    do
        COMPILE=`make BSIZE1D=${LB} BSIZE2D=${BS} BPOWER=${BP} RLEVEL=${N}`
        echo "Compiling with BSIZE2D=$BS RLEVEL=$N"
        echo ${COMPILE}
        echo "DEV=${DEV}  N=${N} B=${BP} R=${N}"
        echo -n "${N}   ${BP}    " >> data/${OUTFILE}_B${BP}_DEV${DEV}.dat
        for q in `seq 1 ${NM}`;
        do
            M=0
            S=0
            # Chosen MAP
            echo "./${BINARY} ${DEV} ${R} ${q} 0.5 1"
            echo -n "[WARMUP] ${METHODS[$(($q-1))]} ($q) map (${SAMPLES} Samples)................"
            x=`${BINARY} ${DEV} ${R} ${q} 0.5 1`
            echo "done"
            echo -n "[BENCHMARK] ${METHODS[$(($q-1))]} ($q) map (${SAMPLES} Samples)............."
            if [ "${BP}" -eq 5 ]; then
                SAMPLES=3
            fi
            for k in `seq 1 ${SAMPLES}`;
            do
                x=`${BINARY} ${DEV} ${R} ${q} 0.5 1`
                oldM=$M;
                M=$(echo "scale=10;  $M+($x-$M)/$k"           | ~/bc)
                S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | ~/bc)
            done
            echo "done"
            MEAN=$M
            VAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"  | ~/bc)
            STDEV=$(echo "scale=10; sqrt(${VAR})"       | ~/bc)
            STERR=$(echo "scale=10; ${STDEV}/sqrt(${SAMPLES})" | ~/bc)
            TMEAN[$q]=${MEAN}
            TVAR[$q]=${VAR}
            TSTDEV[$q]=${STDEV}
            TSTERR[$q]=${STERR}
            echo "---> BPOWER=${BP} RLEVEL=${N} --> (MEAN, VAR, STDEV, STERR) -> (${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]})"
            echo -n "${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]}         " >> data/${OUTFILE}_B${BP}_DEV${DEV}.dat
            echo " "
        done
        echo " " >> data/${OUTFILE}_B${BP}_DEV${DEV}.dat
        echo ""
        echo ""
        echo ""
    done
    echo " "
done
echo "END TIME"
timedatectl
echo "*******************************"
