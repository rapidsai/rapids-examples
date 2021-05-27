
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.types cimport size_type

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

cdef extern from "nvs_udf.cpp":
    pass

# Declare the class with cdef
cdef extern from "nvs_udf.hpp":
    #
    cdef unique_ptr[column] test_parms(string, string, size_type, vector[column_view])
    #
    cdef unique_ptr[column] process_udf(string, string, size_type, vector[column_view])
