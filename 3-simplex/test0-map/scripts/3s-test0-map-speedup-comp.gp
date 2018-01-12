reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/3s-test0-map-speedup-comp.eps'
set title '3s-zero-work Kernel Speedup'
set ytics nomirror
unset ytics
set xtics 256
set xrange [32:1408]
set y2tics 1
set link y2
set y2label 'S_{{/Symbol l}}' rotate by 0 offset 0
set xlabel 'N'
set font "Times, 20"
#set log x 2
#set log y
#set format y2 "10^{%L}"
#set format x "2^{%L}"
set key Left right center reverse samplen 1.0 font "Times,20" spacing 1


set style line 1 dashtype 3 pt 8 lw 1.0 lc rgb "#444444"
set style line 2 dashtype 3 pt 4 lw 1.0 lc rgb "#444444"
set style line 3 dashtype 3 pt 10 lw 1.0 lc rgb "#444444"
set style line 4 dashtype 3 pt 81 lw 1.0 lc rgb "#444444"

# gradiente verdes
set style line 9   dashtype 3 pt 8 lw 3.0 lc rgb "black"
set style line 10  dashtype 2 pt 4 lw 3.0 lc rgb "black"
set style line 11  dashtype 6 pt 10 lw 3.0 lc rgb "black"
set style line 12  dashtype 1 pt 81 lw 3.0 lc rgb "black"

fbb(x)=1
set pointsize   0.5
plot    fbb(x) notitle dashtype 2 lc rgb "black",\
        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "Titan X" with linespoints ls 1,\
        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "GTX 1050 Ti" with linespoints ls 2,\
        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "Tesla K40" with linespoints ls 3,\
        'data/3s-test0-map-titanx_B8.dat' u 1:($3/$7) title "Tesla V100" with linespoints ls 4


