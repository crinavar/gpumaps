reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/write_speedup.eps'
set title 'Single Write, {/Symbol l}({/Symbol w}) Speedup'
set ytics nomirror
unset ytics
set xtics 256
set ytics 1
set link y2
set ylabel 'S_{{/Symbol l}}' rotate by 0 offset 1
set xlabel 'N'
set font "Times, 20"
#set log x 2
#set log y
#set format y2 "10^{%L}"
#set format x "2^{%L}"
set key Right right reverse samplen 1.0 font "Times,20" spacing 1
set style line 1 dashtype 5 pt 1 lw 2.0 lc rgb "#a6a6a6"
set style line 2 dashtype 4 pt 2 lw 2.0 lc rgb "#a6a6a6"
set style line 3 dashtype 6 pt 3 lw 2.0 lc rgb "#a6a6a6"
set style line 4 dashtype 2 pt 4 lw 2.0 lc rgb "#a6a6a6"
set style line 5 dashtype 3 pt 5 lw 2.0 lc rgb "#a6a6a6"
set style line 6 dashtype 3 pt 6 lw 2.0 lc rgb "#a6a6a6"
set style line 7 dashtype 3 pt 7 lw 2.0 lc rgb "#b0b0b0"
set style line 8 dashtype 3 pt 8 lw 2.0 lc rgb "#a6a6a6"

f1(x)=5.9  - b1*(c1/x)**d1
f2(x)=5.86 - b2*(c2/x)**d2
f3(x)=4    - b3*(c3/x)**d3
f4(x)=3.4  - b4*(c4/x)**d4
f5(x)=3    - b5*(c5/x)**d5
f6(x)=2    - b6*(c6/x)**d6
f7(x)=2    - b7*(c7/x)**d7
f8(x)=2    - b8*(c8/x)**d8
fbb(x)=1

fit f1(x) 'data/blockconf_B1.dat' u 1:($3/$7) via b1,c1,d1
fit f2(x) 'data/blockconf_B2.dat' u 1:($3/$7) via b2,c2,d2
fit f3(x) 'data/blockconf_B3.dat' u 1:($3/$7) via b3,c3,d3
fit f4(x) 'data/blockconf_B4.dat' u 1:($3/$7) via b4,c4,d4
fit f5(x) 'data/blockconf_B5.dat' u 1:($3/$7) via b5,c5,d5
fit f6(x) 'data/blockconf_B6.dat' u 1:($3/$7) via b6,c6,d6
fit f7(x) 'data/blockconf_B7.dat' u 1:($3/$7) via b7,c7,d7
fit f8(x) 'data/blockconf_B8.dat' u 1:($3/$7) via b8,c8,d8

set pointsize   0.8
plot    fbb(x) notitle dashtype 2 lc rgb "black",\
        'data/blockconf_B1.dat' u 1:($3/$7) title "{/Symbol r}=1" with points ls 1, f1(x) notitle lw 2.0,\
        'data/blockconf_B2.dat' u 1:($3/$7) title "{/Symbol r}=2" with points ls 2, f2(x) notitle lw 2.0,\
        'data/blockconf_B4.dat' u 1:($3/$7) title "{/Symbol r}=4" with points ls 3, f4(x) notitle lw 2.0,\
        'data/blockconf_B8.dat' u 1:($3/$7) title "{/Symbol r}=8" with points ls 7, f8(x) notitle lw 2.0 
