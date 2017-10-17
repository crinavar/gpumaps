reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/write_running_time.eps'
set title 'Single Write, Running Times'
set ytics nomirror
unset ytics
set xtics 256
set y2tics 10
set link y2
set y2label 'T[ms]' rotate by 0 offset 0

set yrange [0:50]
set xlabel 'N'
set font "Times, 20"
#set log x 2
#set log y
#set format y2 "10^{%L}"
#set format x "2^{%L}"

set pointsize   0.5
set key Right right reverse samplen 3.0 font "Times,20" spacing 1
set style line 1 dashtype 5 lw 2.0 lc rgb "#99d6ff"
set style line 2 dashtype 4 lw 2.0 lc rgb "#66c2ff"
set style line 3 dashtype 6 lw 2.0 lc rgb "#33adff"
set style line 4 dashtype 2 lw 2.0 lc rgb "#0099ff"
set style line 5 dashtype 3 lw 2.0 lc rgb "#007acc"
set style line 6 dashtype 1 pt 9 lw 1.0 lc rgb "#003d66"
set style line 20 lw 1.0 lc rgb "#444964"
set style line 7 dashtype 5 lw 2.0 lc rgb "#9fdf9f"
set style line 8 dashtype 4 lw 2.0 lc rgb "#79d279"
set style line 9 dashtype 6 lw 2.0 lc rgb "#53c653"
set style line 10 dashtype 2 lw 2.0 lc rgb "#39ac39"
set style line 11 dashtype 3 pt 7 lw 1.0 lc rgb "#2d862d"
set style line 12 dashtype 1 pt 7 lw 2.0 lc rgb "#206020"

plot    'data/blockconf_B1.dat' using 1:3 title "BB_{{/Symbol r}=1}" with lines ls 1,\
        'data/blockconf_B2.dat' using 1:3 title "BB_{{/Symbol r}=2}" with lines ls 2,\
        'data/blockconf_B4.dat' using 1:3 title "BB_{{/Symbol r}=4}" with lines ls 3,\
        'data/blockconf_B8.dat' using 1:3 title "BB_{{/Symbol r}=8}" with linespoints ls 6,\
        'data/blockconf_B1.dat' using 1:7 title "{/Symbol l}({/Symbol w})_{{/Symbol r}=1}" with linespoints ls 7,\
        'data/blockconf_B2.dat' using 1:7 title "{/Symbol l}({/Symbol w})_{{/Symbol r}=2}" with linespoints ls 8,\
        'data/blockconf_B4.dat' using 1:7 title "{/Symbol l}({/Symbol w})_{{/Symbol r}=4}" with linespoints ls 9,\
        'data/blockconf_B8.dat' using 1:7 title "{/Symbol l}({/Symbol w})_{{/Symbol r}=8}" with linespoints ls 11,\
