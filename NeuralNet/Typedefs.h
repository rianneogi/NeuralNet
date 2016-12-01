#pragma once

#define EIGEN_USE_BLAS

#include "math.h"
#include <vector>
#include "conio.h"
#include "assert.h"
#include "time.h"
#include <fstream>
#include <boost\numeric\ublas\vector.hpp>
#include <boost\numeric\ublas\matrix.hpp>
#include <boost\numeric\ublas\io.hpp>
#include <Eigen\Dense>

typedef double Float;
typedef Eigen::Matrix<Float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef std::vector<Vector> Dataset;