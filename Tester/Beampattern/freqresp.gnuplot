reset
set xlabel "Arrival Angle (degrees)" font "arial,8"
set ylabel "Frequency (Hz)" font "arial,8"
set zlabel "Gain (dB)" font "arial,8"
set grid lc rgbcolor "#BBBBBB"
set xrange[-90:90]
set yrange[0:10000]
set zrange[-40:0]
unset key
set view 30,56,0.98
splot 'data/freqresp.dat' u 1:2:3 with pm3d
