import cudf
import cudfstrings_udf as udf

s = cudf.Series(["First Last", "James Bond", "Ethan Hunt"])
print(s)

fn = """
__global__ void udf( dstring_view* d_in_strs, dstring* d_out_strs, int count )
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if( tid < count )
    {
        dstring str = d_in_strs[tid];
        dstring names[2];
        str.split( " ", 2, names );
        dstring_view rev_names[2];
        rev_names[0] = names[1];
        rev_names[1] = names[0];
        dstring out(", ",2);
        d_out_strs[tid] = out.join(rev_names,2);
    }
}
"""

su = udf.process_udf(fn, "udf", s._column)
ss = cudf.Series(su)
print(ss)
