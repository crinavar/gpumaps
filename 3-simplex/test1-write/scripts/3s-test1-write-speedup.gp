reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/3s-test1-write-speedup.eps'
set title '3-simplex, {/Symbol l}({/Symbol w}) Speedup on Single Write'
set ytics nomirror
unset ytics
set xtics 256
#set ytics 1
set link y2
set y2tics 1
set y2label 'S_{{/Symbol l}}' rotate by 0 offset 0
set y2range [0:6]
set xlabel 'N'
set font "Times, 20"
#set log x 2
#set log y
#set format y2 "10^{%L}"
#set format x "2^{%L}"
#set key Right right center reverse samplen 1.0 font "Times,20" spacing 1
set key at 1400,5.5 reverse samplen 1.0 font "Times,20" spacing 1


set style line 1 dashtype 3 pt 8 lw 1.0 lc rgb "#222222"
set style line 2 dashtype 3 pt 4 lw 1.0 lc rgb "#222222"
set style line 3 dashtype 3 pt 10 lw 1.0 lc rgb "#222222"
set style line 4 dashtype 3 pt 81 lw 1.0 lc rgb "#222222"

# gradiente verdes
set style line 9   dashtype 3 pt 8 lw 3.0 lc rgb "black"
set style line 10  dashtype 2 pt 4 lw 3.0 lc rgb "black"
set style line 11  dashtype 6 pt 10 lw 3.0 lc rgb "black"
set style line 12  dashtype 1 pt 81 lw 3.0 lc rgb "black"

f1(x)=5.9  - b1*(c1/x)**d1
f2(x)=5.86 - b2*(c2/x)**d2
f3(x)=4    - b3*(c3/x)**d3
f4(x)=3.4  - b4*(c4/x)**d4
f5(x)=3    - b5*(c5/x)**d5
f6(x)=2    - b6*(c6/x)**d6
f7(x)=2    - b7*(c7/x)**d7
f8(x)=2    - b8*(c8/x)**d8
fbb(x)=1

fit f1(x) 'data/3s-test1-write-titanx_B1.dat' u 1:($3/$7) via b1,c1,d1
fit f2(x) 'data/3s-test1-write-titanx_B2.dat' u 1:($3/$7) via b2,c2,d2
fit f3(x) 'data/3s-test1-write-titanx_B3.dat' u 1:($3/$7) via b3,c3,d3
fit f4(x) 'data/3s-test1-write-titanx_B4.dat' u 1:($3/$7) via b4,c4,d4
fit f5(x) 'data/3s-test1-write-titanx_B5.dat' u 1:($3/$7) via b5,c5,d5
fit f6(x) 'data/3s-test1-write-titanx_B6.dat' u 1:($3/$7) via b6,c6,d6
fit f7(x) 'data/3s-test1-write-titanx_B7.dat' u 1:($3/$7) via b7,c7,d7
fit f8(x) 'data/3s-test1-write-titanx_B8.dat' u 1:($3/$7) via b8,c8,d8

set pointsize   0.8
plot    fbb(x) notitle dashtype 2 lc rgb "black",\
        'data/3s-test1-write-titanx_B1.dat' u 1:($3/$7) title "{/Symbol r}=1" with points ls 1, f1(x) notitle ls 9,\
        'data/3s-test1-write-titanx_B2.dat' u 1:($3/$7) title "{/Symbol r}=2" with points ls 2, f2(x) notitle ls 10,\
        'data/3s-test1-write-titanx_B4.dat' u 1:($3/$7) title "{/Symbol r}=4" with points ls 3, f4(x) notitle ls 11,\
        'data/3s-test1-write-titanx_B8.dat' u 1:($3/$7) title "{/Symbol r}=8" with points ls 4, f8(x) notitle ls 12
