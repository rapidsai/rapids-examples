# libcuml and libcudf groupby

This C++ example demonstrates running a `libcuml` regression model on groups from a `libcudf` `groupby` operation

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/libcudf_example
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
