%module cfrechet

%{
#define SWIG_FILE_WITH_INIT
#include "cfrechet.h"
%}

%include "numpy.i"

%init %{
import_array();
%}
%apply (double* IN_ARRAY1, int DIM1) {(double* P, int D1)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Q, int D2)};
%apply (double* IN_ARRAY1, int DIM1) {(double* ca, int D3)};
%include "cfrechet.h"

