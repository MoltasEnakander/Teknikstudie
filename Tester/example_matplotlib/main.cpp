#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main()
{
	std::vector<double> x, y;
	x.push_back(10);
	x.push_back(12);
	x.push_back(14);
	x.push_back(18);
	y.push_back(1);
	y.push_back(4);
	y.push_back(12);
	y.push_back(25);
	auto h = plt::figure(1);	
	auto ax1 = h.add_subplot(211);
	//ax1.xlim(-60 ,60);
	plt::xlim(-60, 60);
	plt::ylim(-60, 60);
	plt::grid(true);
	plt::title("Standard usage"); // set a title
	plt::xlabel("theta");
	plt::ylabel("phi");	
	plt::plot(x,y);
	plt::show();

	return 0;
}

/*#define _USE_MATH_DEFINES
#include "matplotlibcpp.h"
#include <cmath>

namespace plt = matplotlibcpp;

int main() {
  int n = 1000;
  std::vector<double> x, y, z;

  for (int i = 0; i < n; i++) {
    x.push_back(i * i);
    y.push_back(sin(2 * M_PI * i / 360.0));
    z.push_back(log(i));

    if (i % 10 == 0) {
      // Clear previous plot
      plt::clf();
      // Plot line from given x and y data. Color is selected automatically.
      plt::plot(x, y);
      // Plot a line whose name will show up as "log(x)" in the legend.
      plt::plot(x, z, {{"label", "log(x)"}});

      // Set x-axis to interval [0,1000000]
      plt::xlim(0, n * n);

      // Add graph title
      plt::title("Sample figure");
      // Enable legend.
      plt::legend();
      // Display plot continuously
      plt::pause(0.01);
    }
  }
}*/