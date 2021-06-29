import cudf, dask_cudf
from dask import delayed
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_metrics import Monitor


cluster = LocalCUDACluster()
client = Client()

# create monitor bound to client
monitor = Monitor(client)
# attach to cluster and start collecting metrics with provided tracking list
monitor.attach(tracking=['total-mem', 'mem-util', 'compute-util'])
monitor.start()
client.close() # close client used to start metrics

# generate some random CSV files
df = cudf.datasets.timeseries(freq='6H')
df.to_csv('testing/small_file.csv')

df = cudf.datasets.timeseries()
df.to_csv('testing/big_file.csv')


# dag 1 -- should be low utilization
fut = delayed(dask_cudf.read_csv)('testing/small_file.csv')
fut = delayed(fut.to_parquet)('testing/small_file.parquet')

client = Client(cluster) # new client
fut = client.compute(fut) # submit job to cluster
client.gather(fut) # wait for it to finish

monitor = Monitor(client)
monitor.stop() # stop recording
monitor.to_csv('testing/metrics/small') # export metrics to csv
monitor.clear_metrics() # clear metrics before next job


# dag 2 -- should be better utilization
fut = delayed(dask_cudf.read_csv)('testing/big_file.csv')
fut = delayed(fut.to_parquet)('testing/big_file.parquet')

monitor.start() # start recording again
fut = client.compute(fut)
client.gather(fut)
monitor.stop()

monitor.to_csv('testing/metrics/big')
for i in monitor.peak_memory():
    print(i) # get peak memory for each worker during last job

client.shutdown()