#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

std::unique_ptr<cudf::table> cuml_regression_on_groupby(cudf::table_view stock_info_table)
{
  // Schema: | Timestamp | Name | X | Y
  auto keys = cudf::table_view{{stock_info_table.column(1)}};  // name

  // Compute the average of each company's closing price with entire column
  cudf::groupby::groupby grpby_obj(keys);

  cudf::groupby::groupby::groups gb_groups = grpby_obj.get_groups(stock_info_table.select({2,3}));
  return std::make_unique<cudf::table>(stock_info_table.select({2,3}));
}

int main(int argc, char** argv)
{
  // Read data
  auto stock_table_with_metadata = read_csv("test.csv");

  // Process
  auto result = cuml_regression_on_groupby(*stock_table_with_metadata.tbl);

  // Write out result
  write_csv(*result, "test_out.csv");

  return 0;
}
