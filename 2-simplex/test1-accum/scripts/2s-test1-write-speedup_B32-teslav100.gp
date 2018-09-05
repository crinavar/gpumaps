reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/2s-test1-write-speedup-B32-teslav100.eps'
set title '2s-write Kernel Speedup, Tesla V100, {/Symbol r}=32'
#set ytics mirror
unset ytics
set xtics (1024, 8192, 16384, 24576, 32768)
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
        'data/2s-test1-write-teslav100_B32.dat' using 1:($3/$27) title "Rectangle" with linespoints ls 2,\
        'data/2s-test1-write-teslav100_B32.dat' using 1:($3/$23) title "{/Symbol l}({/Symbol w})" with linespoints ls 1,\
        'data/2s-test1-write-teslav100_B32-recursive.dat' using 1:($3/$31) title "Recursive" with linespoints ls 3,\
        'data/2s-test1-write-teslav100_B32.dat' using 1:($3/$7) title "UTM" with linespoints ls 4
