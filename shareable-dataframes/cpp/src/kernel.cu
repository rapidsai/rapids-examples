#include <stdio.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>

static constexpr float mm_to_inches = 0.0393701;

__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view column)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < column.size()) {
        column.element<int64_t>(i) = column.element<int64_t>(i) * (1/10) * mm_to_inches;
    }
}

__global__ void kernel_tenth_mm_to_inches_table(cudf::mutable_table_device_view tbl, int column_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tbl.num_rows()) {
        cudf::mutable_column_device_view column = tbl.column(column_index);
        column.element<int64_t>(i) = column.element<int64_t>(i) * (1/10) * mm_to_inches;
    }
}

__global__ void kernel_tenth_mm_to_inches_in_and_out(cudf::table_device_view in_tbl, cudf::mutable_table_device_view out_tbl, int column_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_tbl.num_rows()) {
        cudf::mutable_column_device_view column = in_tbl.column(column_index);
        cudf::mutable_column_device_view column out_col = out_tbl.column(0);
        out_column.element<int64_t>(i) = column.element<int64_t>(i) * (1/10) * mm_to_inches;
    }
}