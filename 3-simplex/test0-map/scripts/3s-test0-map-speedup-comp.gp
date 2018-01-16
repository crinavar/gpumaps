reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/3s-test0-map-speedup-comp.eps'
set title '3s-zero-work Kernel Speedup with {{/Symbol l}}'
set ytics nomirror
unset ytics
set xtics (0, 256, 512, 768, 1024, 1408)
set xrange [32:1408]
set y2range [0.6:1.5]
set y2tics 0.1
set link y2
set y2label 'S_{{/Symbol l}}' rotate by 0 offset 0
set xlabel 'N'
set font "Times, 20"
#set log x 2
#set log y
#set format y2 "10^{%L}"
#set format x "2^{%L}"
set key Left right center reverse samplen 1.0 font "Times,20" spacing 1


set style line 1 dashtype 3 pt 8 lw 1.0 lc rgb "black"
set style line 2 dashtype 3 pt 4 lw 1.0 lc rgb "magenta"
set style line 3 dashtype 3 pt 10 lw 1.0 lc rgb "red"
set style line 4 dashtype 3 pt 81 lw 1.0 lc rgb "forest-green"

# gradiente verdes
set style line 9   dashtype 3 pt 8 lw 3.0 lc rgb "black"
set style line 10  dashtype 2 pt 4 lw 3.0 lc rgb "black"
set style line 11  dashtype 6 pt 10 lw 3.0 lc rgb "black"
set style line 12  dashtype 1 pt 81 lw 3.0 lc rgb "black"

fbb(x)      =1
titanx(x)   =1.5 - b1*(c1/x)**d1
gtx1050ti(x)=1.5 - b2*(c2/x)**d2
teslak40(x) =1.5 - b3*(c3/x)**d3
teslav100(x)=1.5 - b4*(c4/x)**d4

fit titanx(x) 'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) via b1,c1,d1
fit gtx1050ti(x) 'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) via b2,c2,d2
fit teslak40(x) 'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) via b3,c3,d3
fit teslav100(x) 'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) via b4,c4,d4

set pointsize   0.7
#plot    fbb(x) notitle dashtype 2 lc rgb "black",\
#        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "Titan X"       with points ls 1,   titanx(x) notitle ls 9,\
#        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "GTX 1050 Ti"   with points ls 2,   gtx1050ti(x) notitle ls 10,\
#        'data/3s-test0-map-teslak40_B8.dat' u 1:($3/$7) title "Tesla K40"     with points ls 3,   teslak40(x) notitle ls 11,\
#        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "Tesla V100"    with points ls 4,   teslav100(x) notitle ls 12

plot    fbb(x) notitle dashtype 2 lc rgb "black",\
        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "Titan X"       with linespoints ls 1,\
        'data/3s-test0-map-gtx1050ti_B8.dat' u 1:($3/$7) title "GTX 1050 Ti"   with linespoints ls 2,\
        'data/3s-test0-map-teslak40_B8.dat' u 1:($3/$7) title "Tesla K40"     with linespoints ls 3,\
        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "Tesla V100"    with linespoints ls 4
