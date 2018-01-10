reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/test3-ca3d-speedup.eps'
set title '3-Simplex, {/Symbol l}({/Symbol w}) Speedup on CA Rule 5766 Test,({/Symbol r}=8)'
#set ytics nomirror
#set ytics 0.1
unset ytics
set xtics (0, 256, 512, 768, 1024, 1408)
set xrange [0:1408]
set yrange [0.7:1.2]
set link y2
set y2tics 0.1
set y2label 'S_{{/Symbol l}}' rotate by 0 offset -1
set xlabel 'N'
set font "Times, 20"
set key Left left reverse samplen 0.2 font "Times,18" spacing 1

set style line 1 dashtype 3 pt 7 lw 2.0 lc rgb "#222222"
set style line 2 dashtype 3 pt 3 lw 2.0 lc rgb "#222222"
set style line 3 dashtype 3 pt 8 lw 2.0 lc rgb "#222222"

f1(x)=1.13  - b1*(c1/x)**d1
f2(x)=1.09  - b2*(c2/x)**d2
f3(x)=1.5  - b3*(c3/x)**d3
fbb(x)=1

fit f1(x) 'data/3s-test3-ca3d-titanx_B8.dat' u 1:($3/$7) via b1,c1,d1
fit f2(x) 'data/3s-test3-ca3d-gtx1050ti_B8.dat' u 1:($3/$7) via b2,c2,d2
fit f3(x) 'data/3s-test3-ca3d-teslak40_B8.dat' u 1:($3/$7) via b3,c3,d3

set pointsize   0.5
plot    fbb(x) notitle dashtype 2 lc rgb "black",\
        'data/3s-test3-ca3d-titanx_B8.dat' u 1:($3/$7) title "Titan X, GP102, {/Symbol r}=8" with points ls 1, f1(x) notitle lw 2.0 lc rgb "black" dt 1,\
        'data/3s-test3-ca3d-gtx1050ti_B8.dat' u 1:($3/$7) title "GTX 1050Ti, GP107, {/Symbol r}=8" with points ls 2, f2(x) notitle lw 2.0 lc rgb "magenta",\
        'data/3s-test3-ca3d-teslak40_B8.dat' u 1:($3/$7) title "Tesla K40, GK110, {/Symbol r}=8" with points ls 3, f3(x) notitle lw 2.0 lc rgb "red"
