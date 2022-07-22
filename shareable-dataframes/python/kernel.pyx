import cudf
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport mutable_column_view
from cudf._lib.cpp.table.table_view cimport mutable_table_view
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "kernel_wrapper.hpp":
    cdef cppclass C_CudfWrapper "CudfWrapper":
        C_CudfWrapper(mutable_table_view tbl)
        void tenth_mm_to_inches(int column_index)

cdef class CudfWrapper:
    cdef C_CudfWrapper* gdf

    def __cinit__(self, columns):
        cdef vector[mutable_column_view] column_views

        cdef Column col
        for col in columns:
            column_views.push_back(col.mutable_view())
            
        cdef mutable_table_view tv = mutable_table_view(column_views)
        self.gdf = new C_CudfWrapper(tv)

    def cython_tenth_mm_to_inches(self, col_index):
        self.gdf.tenth_mm_to_inches(col_index)
