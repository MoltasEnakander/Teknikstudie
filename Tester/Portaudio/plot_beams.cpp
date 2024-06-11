#include <iostream>
#include <fstream>
#include <string>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main()
{
	const int NUM_BEAMS = 13;
	const int NUM_FILTERS = 6;
	float beams[NUM_BEAMS * NUM_BEAMS * NUM_FILTERS];
	std::string line;
	std::ifstream myfile ("beamsallmixed.txt");
	if (myfile.is_open())
	{
		int i = 0;
		while ( getline (myfile,line) )
		{
			beams[i] = std::stof(line);
			i++;
		}
		myfile.close();
	}

	PyObject *mat1, *mat2, *mat3, *mat4, *mat5, *mat6;

	plt::figure(100);
    plt::clf();
    plt::imshow(&(beams[0]), NUM_BEAMS, NUM_BEAMS, 1, {}, &mat1);
    plt::colorbar(mat1);

    plt::figure(101);
    plt::clf();
    plt::imshow(&(beams[NUM_BEAMS * NUM_BEAMS]), NUM_BEAMS, NUM_BEAMS, 1, {}, &mat2);
    plt::colorbar(mat2);

    plt::figure(102);
    plt::clf();
    plt::imshow(&(beams[2 * NUM_BEAMS * NUM_BEAMS]), NUM_BEAMS, NUM_BEAMS, 1, {}, &mat3);
    plt::colorbar(mat3);

    plt::figure(103);
    plt::clf();
    plt::imshow(&(beams[3 * NUM_BEAMS * NUM_BEAMS]), NUM_BEAMS, NUM_BEAMS, 1, {}, &mat4);
    plt::colorbar(mat4);

    plt::figure(104);
    plt::clf();
    plt::imshow(&(beams[4 * NUM_BEAMS * NUM_BEAMS]), NUM_BEAMS, NUM_BEAMS, 1, {}, &mat5);
    plt::colorbar(mat5);

    plt::figure(105);
    plt::clf();
    plt::imshow(&(beams[5 * NUM_BEAMS * NUM_BEAMS]), NUM_BEAMS, NUM_BEAMS, 1, {}, &mat6);
    plt::colorbar(mat6);

    plt::show();

    return 0;
}