set key noautotitle
set title "Up Boson Density"
set title  font ", 20" norotate
set xlabel "x"
set ylabel "y"
set xlabel font ", 20" norotate
set ylabel font ", 20" norotate
set xtics font ",20"
set ytics font ",20"
set xrange[0:16]
set yrange[0:16]
set cbrange [0:65]
set terminal png size 400,300 enhanced font "Helvetica,20"
set output 'foo.png'
p 'bar.dat' u 1:2:3 w image 

pause mouse close
