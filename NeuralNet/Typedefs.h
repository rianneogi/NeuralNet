#pragma once

#define EIGEN_USE_BLAS

#include "math.h"
#include <vector>
#include <iostream>
#include "conio.h"
#include "assert.h"
#include "time.h"
#include <fstream>

//#include <clBLAS.h>
//#include <cblas.h>

#include <Eigen\Dense>

//#include <boost\numeric\ublas\vector.hpp>
//#include <boost\numeric\ublas\matrix.hpp>
//#include <boost\numeric\ublas\io.hpp>

//#include <armadillo>

//#include <unsupported\Eigen\CXX11\src\Tensor\Tensor.h>

typedef double Float;
typedef Eigen::Matrix<Float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

//typedef arma::vec Vector;
//typedef arma::mat Matrix;

typedef std::vector<Vector> Dataset;