reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/3s-test0-map-rt.eps'
set title '3-simplex, {/Symbol l}({/Symbol w}) running times for Zero-Work Test'
set ytics nomirror
unset ytics
set xtics 256
set ytics
set xrange [32:1408]
set log y
set ylabel 'T[s]' rotate by 0 offset 2
set xlabel 'N'
set font "Times, 20"
set format y "10^{%L}"
#set format x "2^{%L}"
set key Right right bottom reverse samplen 2.0 font "Times,20" spacing 1

# gradiente verdes
set style line 7   dashtype 5 pt 2 lw 2.0 lc rgb "#9fdf9f"
set style line 8   dashtype 4 pt 3 lw 2.0 lc rgb "#79d279"
set style line 9   dashtype 3 pt 8 lw 2.0 lc rgb "#53c653"
set style line 10  dashtype 2 pt 4 lw 2.0 lc rgb "#39ac39"
set style line 11  dashtype 4 pt 10 lw 2.0 lc rgb "#2d862d"
set style line 12  dashtype 1 pt 81 lw 2.0 lc rgb "#206020"

set pointsize   0.7
plot    'data/3s-test0-map-titanx_B1.dat' u 1:7 title "{/Symbol r}=1" with lines ls 9,\
        'data/3s-test0-map-titanx_B2.dat' u 1:7 title "{/Symbol r}=2" with lines ls 10,\
        'data/3s-test0-map-titanx_B4.dat' u 1:7 title "{/Symbol r}=4" with lines ls 11,\
        'data/3s-test0-map-titanx_B8.dat' u 1:7 title "{/Symbol r}=8" with lines ls 12


