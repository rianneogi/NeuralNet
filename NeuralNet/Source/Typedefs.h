#pragma once

//#define NN_DEBUG
//#define USE_GPU

#include "math.h"
#include <vector>
#include <iostream>
#include "conio.h"
#include "assert.h"
#include "time.h"
#include <fstream>

#include <clBLAS.h>
#include <cblas.h>

#ifdef USE_GPU
typedef cl_float Float;
#else
typedef double Float;
#endif