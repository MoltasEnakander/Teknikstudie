/*#include <iostream>
#include <vector>
#include <include/pybind11/pybind11.h>
#include <include/pybind11/embed.h>  // python interpreter
#include <include/pybind11/stl.h>  // type conversion

namespace py = pybind11;

int main() {
  std::cout << "Starting pybind" << std::endl;
  py::scoped_interpreter guard{}; // start interpreter, dies when out of scope

  py::function min_rosen =
      py::reinterpret_borrow<py::function>(   // cast from 'object' to 'function - use `borrow` (copy) or `steal` (move)
          py::module::import("exec_numpy").attr("min_rosen")  // import method "min_rosen" from python "module"
      );

  py::function my_func =
      py::reinterpret_borrow<py::function>(
          py::module::import("exec_numpy").attr("my_func")  
      );

  //py::object result = min_rosen(std::vector<double>{1,2,3,4,5});  // automatic conversion from `std::vector` to `numpy.array`, imported in `pybind11/stl.h`
  py::list result = my_func(std::vector<double>{1,2,3,4,5});  // automatic conversion from `std::vector` to `numpy.array`, imported in `pybind11/stl.h`
  //bool success = result.attr("success").cast<bool>();
  //int num_iters = result.attr("nit").cast<int>();
  //printf("%d\n", num_iters);
  py::list list_values = result.attr("fun").cast<py::list>();
  printf("%f\n", list_values[0].cast<double>());
}*/


#include <include/pybind11/pybind11.h>
#include <include/pybind11/embed.h>  // python interpreter
#include <include/pybind11/stl.h>  // type conversion
#include <iostream>

#include <chrono>
#include <ctime>
#include <unistd.h>

namespace py = pybind11;

int main() {
    py::scoped_interpreter python;

    py::function my_func =
      py::reinterpret_borrow<py::function>(
          py::module::import("filtercreation").attr("filtercreation")  
      );
    
    //auto scipy = py::module::import("scipy.signal");
    //py::list res = scipy.attr("firwin")(10, std::vector<float>{0.3f});
    py::list res = my_func(3, 9, 0.05);
    for (py::handle obj : res) {  // iterators!
        std::cout << "  - " << res[0].attr("__float__")().cast<float>() << std::endl;
    }

    int j;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < 16*6*2048; ++i)
    {
      j = 0;
    }


    
    
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";
}