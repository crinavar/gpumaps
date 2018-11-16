reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 20
set output 'plots/2s-test1-accum-speedup-TITANV.eps'
set title '2s-accum Kernel Speedup, TITAN V, {/Symbol r}=16' font "Times, 24"
#set ytics mirror
unset ytics
set xtics (1024, 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536)
set y2tics 0.2
set link y2
#set logscale y
#set logscale x
#set logscale y2
set y2label 'S_{{/Symbol l}}' rotate by 0 offset -1

#set yrange [0.15:1.2]
set xlabel 'N'
set font "Times, 20"
#set log x 2
#set format y "10^{%L}"
#set format x "2^{%L}"

set pointsize   0.5
set key Left bottom right reverse samplen 3.0 font "Times,20" spacing 1

set style line 1 dashtype 1 pt 7 lw 1.0 lc rgb "#2d905d"
set style line 2 dashtype 1 pt 9 lw 1.0 lc rgb "magenta"
set style line 3 dashtype 1 pt 5 lw 1.0 lc rgb "#1E90FF"
set style line 4 dashtype 1 pt 2 lw 1.0 lc rgb "red"

fbb(x)=1
plot    fbb(x) notitle dashtype 2 lc rgb "black",\
        'data/2s-test1-accum-TITANV_B16.dat' using 1:($3/$7)   title "{/Symbol l}({/Symbol w})" with lines ls 1,\
        'data/2s-test1-accum-TITANV_B16.dat' using 1:($3/$11)  title "Rectangle" with lines ls 2,\
        'data/2s-test1-accum-TITANV_B16.dat' using 1:($3/$15)  title "Trapezoid"                with lines ls 3
