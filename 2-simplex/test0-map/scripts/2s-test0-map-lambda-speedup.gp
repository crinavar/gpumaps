reset
set macro
red_000 = "#F9B7B0"
red_025 = "#F97A6D"
red_050 = "#E62B17"
red_075 = "#8F463F"
red_100 = "#6D0D03"

blue_000 = "#A9BDE6"
blue_025 = "#7297E6"
blue_050 = "#1D4599"
blue_075 = "#2F3F60"
blue_100 = "#031A49"

green_000 = "#A6EBB5"
green_025 = "#67EB84"
green_050 = "#11AD34"
green_075 = "#2F6C3D"
green_100 = "#025214"

brown_000 = "#F9E0B0"
brown_025 = "#F9C96D"
brown_050 = "#E69F17"
brown_075 = "#8F743F"
brown_100 = "#6D4903"
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/2s-test0-zerowork-lambda-speedup.eps'
set title '2s-zero-work, {{/Symbol l}} Mapping Speedup, CUDA sqrtf()'
set ytics mirror
#unset ytics
set xtics (1024, 8192, 16384, 24576, 32768)
#set ytics 0.1
#set logscale y
#set logscale x
#set logscale y2
set ylabel 'S' rotate by 0 offset 2

#set yrange [1.96:2]
#set format y "10^{%L}"
set xlabel 'N'
set font "Times, 20"

set pointsize 0.5
set key Left bottom right reverse samplen 3.0 font "Times,20" spacing 1

# Newton
set style line 1 dashtype 5 lw 1.0 pt 7 lc rgb blue_025
# CUDA
set style line 2 dashtype 1 lw 1.0 pt 1 lc rgb "black"
# Inverse
set style line 3 dashtype 6 lw 1.0 pt 9 lc rgb red_050

set style line 7   dashtype 5 pt 2 lw 1.0 lc rgb "#9fdf9f"
set style line 8   dashtype 4 pt 3 lw 1.0 lc rgb "#79d279"
set style line 9   dashtype 6      lw 1.0 lc rgb "#53c653"
set style line 10  dashtype 2      lw 2.0 lc rgb "#39ac39"
set style line 11  dashtype 3 pt 7 lw 1.0 lc rgb "#2d862d"
set style line 12  dashtype 1 pt 7 lw 1.0 lc rgb "#206020"

plot    'data/2s-test0-map-titanx_B1.dat' using 1:($3/$23) title "{{/Symbol r}} = 1" with lines ls 7,\
        'data/2s-test0-map-titanx_B2.dat' using 1:($3/$23) title "{{/Symbol r}} = 2" with lines ls 8,\
        'data/2s-test0-map-titanx_B4.dat' using 1:($3/$23) title "{{/Symbol r}} = 4" with lines ls 9,\
        'data/2s-test0-map-titanx_B8.dat' using 1:($3/$23) title "{{/Symbol r}} = 8" with lines ls 10,\
        'data/2s-test0-map-titanx_B16.dat' using 1:($3/$23) title "{{/Symbol r}} = 16" with lines ls 11 lw 2.0,\
        'data/2s-test0-map-titanx_B32.dat' using 1:($3/$23) title "{{/Symbol r}} = 32" with lines ls 12 lw 2.0,\
