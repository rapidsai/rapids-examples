# dask-metrics

A tool for collecting metrics about the performance and utilization of GPUs on distributed Dask clusters.

## Overview

dask-metrics will collect information about how your cluster is performing in the background with little to no management necessary and minimal overhead.

Existing workflows do not need to be modified to take advantage of dask-metrics. Simply connect to the cluster and start the monitor before submitting jobs and then stop the monitor when you're done.

After metrics have been collected, the user can use dask-metrics to calculate statistics like peak memory usage or they can export those metrics to csv files to be processed and examined elsewhere.

## Metrics Tracked

### Always Tracked
* **`job`**: the job that is currently executing.
* **`dag`**: the dag within the job that is currently executing.
* **`timestamp`**: the time, in seconds, since the start of metric collection.

### Optional
* **`total-mem`**: the memory used, in bytes, by each GPU on a worker
* **`mem-util`**: the memory utilization of each GPU on a worker as a percentage of total memory available.
* **`compute-util`**: the compute utilization of each GPU on a worker as a percentage of maximum compute ability.

## How to Use

Existing workflows to not need to be modified to use dask-metrics. The only code that needs to be added is at the beginning and end of your workflow.

Before the jobs you want to monitor are submitted to the cluster, first attach a `Monitor` to a Client object, pass it a list of metrics to track, and start it.
```python
from dask_metrics import Monitor

client = Client('[scheduler address]')  # Client connected to cluster
monitor = Monitor(client)  # create Monitor object bound to client

# attach monitor to cluster with list of metrics to track
monitor.attach(['mem-util', 'compute-util'])
monitor.start()  # start recording metrics

client.close()
```

After the monitor is attached and started, run any jobs just as you normally would.

After you are done running everything, connect to the cluster once more to stop the monitor and export the metrics.
```python
from dask_metrics import Monitor

client = Client('[scheduler address]')  # Client connected to cluster
monitor = Monitor(client)  # Monitor object bound to client

monitor.stop()  # stop recording metrics
monitor.to_csv('path/to/folder')  # export metrics to folder

client.close()
```

Important to note is that `to_csv` will create a csv file in the folder specified for *each* worker in the cluster.

### Peak Memory

If you want information about the peak memory utilization across the workers in your cluster, you can use `worker.peak_memory()` to get a list of workers, where each worker is represented by a list of the peak memory utilizations of the compute devices on that worker.

As an example, the return value for a cluster with 2 workers and 4 GPUs on each worker would be in this format: `[[12, 8, 46, 27], [4, 0, 9, 13]]`

## Additional Information

Currently, in the case that a task submits another task to the cluster during its execution, that task will be considered part of the first when counting job and dag numbers.

Please do not hesitate to report any bugs you might find or suggest useful features.
