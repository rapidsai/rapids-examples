# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column

from nvs_udf cimport (
    test_parms as cpp_test_parms,
    process_udf as cpp_process_udf
)

def process_udf( udf, name, Column scol ):
    cdef string c_udf
    cdef string c_name
    cdef column_view c_view
    cdef size_type c_size
    cdef unique_ptr[column] c_result

    cdef vector[column_view] c_columns

    c_udf = udf.encode('UTF-8')
    c_name = name.encode('UTF-8')
    c_size = scol.size
    c_view = scol.view()
    c_columns.push_back(c_view)

    #with nogil:
    c_result = move(cpp_process_udf(c_udf, c_name, c_size, c_columns))

    return Column.from_unique_ptr(move(c_result))
