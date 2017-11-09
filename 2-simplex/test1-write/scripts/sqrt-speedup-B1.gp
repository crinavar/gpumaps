reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/2s-sqrt-speedup-B1.eps'
set title '2-Simplex, Square Root Speedup, {/Symbol r}=1'
set ytics mirror
#unset ytics
set xtics (1024, 8192, 16384, 24576, 32768)
set ytics 0.1
#set logscale y
#set logscale x
#set logscale y2
set ylabel 'S_{/Symbol l}' rotate by 0 offset 2

#set yrange [1.96:2]
set xlabel 'N'
set font "Times, 20"
#set log x 2
#set format y "10^{%L}"
#set format x "2^{%L}"

set pointsize   0.5
set key Left bottom right reverse samplen 3.0 font "Times,20" spacing 1

# BB
set style line 1 dashtype 5 lw 1.0 lc rgb "#99d6ff"
set style line 2 dashtype 4 lw 1.0 lc rgb "#66c2ff"
set style line 3 dashtype 6 lw 1.0 lc rgb "#33adff"
set style line 4 dashtype 2 lw 1.0 lc rgb "#0099ff"
set style line 5 dashtype 3 lw 1.0 lc rgb "#007acc"
set style line 6 dashtype 1 pt 9 lw 1.0 lc rgb "#259bff"
set style line 20 lw 1.0 lc rgb "#444964"

# Lambda
set style line 7   dashtype 5 pt 2 lw 1.0 lc rgb "#9fdf9f"
set style line 8   dashtype 4 pt 3 lw 1.0 lc rgb "#79d279"
set style line 9   dashtype 6      lw 1.0 lc rgb "#53c653"
set style line 10  dashtype 2      lw 2.0 lc rgb "#39ac39"
set style line 11  dashtype 3 pt 7 lw 1.0 lc rgb "#2d862d"
set style line 12  dashtype 1 pt 7 lw 1.0 lc rgb "#206020"

plot    'data/test1_B1.dat' using 1:($3/$11) title "Newton_{{/Symbol r}=1}" with lines ls 1,\
        'data/test1_B1.dat' using 1:($3/$15) title "Default_{{/Symbol r}=2}" with lines ls 2,\
        'data/test1_B1.dat' using 1:($3/$23) title "Inverse_{{/Symbol r}=4}" with lines ls 3
