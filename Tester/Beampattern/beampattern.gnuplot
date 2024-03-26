reset
unset key
set xlabel "Arrival Angle (degrees)" font "arial,12"
set ylabel "Gain (dB)" font "arial,12"
set grid lc rgbcolor "#BBBBBB"
plot 'data/beampattern.dat' u 2:5 w l
