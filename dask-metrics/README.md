# dask-metrics

A tool for collecting metrics about the performance and utilization of GPUs on distributed Dask clusters.

## Overview

dask-metrics will collect information about how your cluster is performing in the background with little to no management necessary and minimal overhead.

Existing workflows do not need to be modified to take advantage of dask-metrics. Simply connect to the cluster and start the monitor before submitting jobs and then stop the monitor when you're done.

After metrics have been collected, the user can use dask-metrics to calculate statistics like peak memory usage or they can export those metrics to csv files to be processed and examined elsewhere.

## Installation

Install through conda:

```bash
conda install dask-metrics -c travishester
```

Or install through pip:

```bash
pip install .
```

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
monitor.attach(tracking=['mem-util', 'compute-util'])
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

### Monitor.attach Parameters

Parameters passed to `Monitor.attach` to configure behavior.

#### Required:

* **`tracking`**: list of metrics to track during run

#### Optional:

* [**`custom_metrics`**](#custom-metrics): list of custom metric functions for the monitor to track
* [**`job_type` (default 'client')**](#job-number-tracking): determines the logic for tracking job numbers.
* **`polling_interval` (default 200)**: number of milliseconds to wait between collecting new measurements.
* **`mem_limit`**: If set, each worker will dump metrics onto disk after `mem_limit` number of measurements to save local memory.
* **`dump_loc` (default 'temp')**: Location on disk that each worker will store temporary metrics if `mem_limit` is set.

### Job Number Tracking

The default logic for tracking job numbers is client-based (`job_type='client'`). This means that a job consists of the dags submitted from a single client connection. A new client connection would be a new job.

You can also enable manual job number tracking (`job_type='manual'`). In this configuration, job numbers are independent of client connections. The only way to increment the job number is with a call to `new_job()` from a Monitor object. You can pass it the named argument `job` to specify a specific job number instead of just incrementing it by 1 as well.

### Peak Memory

If you want information about the peak memory utilization across the workers in your cluster, you can use `worker.peak_memory()` to get a list of workers, where each worker is represented by a list of the peak memory utilizations of the compute devices on that worker.

As an example, the return value for a cluster with 2 workers and 4 GPUs on each worker would be in this format: `[[12, 8, 46, 27], [4, 0, 9, 13]]`

## Visualization

Included also are tools for visualizing the metrics you collect.

There are two basic plot types: lines and boxes. Shown below is the basic syntax for plotting a job on a single worker as a line graph.

```python
from dask_metrics import visualize as vis
job_number = 1
metrics = ['compute-util', 'mem-util']
vis.lines('path/to/worker/file.csv', job_number, metrics)
```

If you wanted to visualize your metrics with a box an whisker plot, you would simply use `vis.boxes` instead.

Both plot types take the following parameters:
* **`worker_file`**: The path to the csv output for the worker you wish to visualize
* **`job`**: The number of the job you want to visualize
* [**`metrics`**](#metrics-tracked): The list of metric names you want to plot
* **`width` (default 20)**: The width in inches of the drawn plot
* **`height` (default 10)**: The height in inches of the drawn plot
* **`save` (optional)**: The path to the image file you wish to save the plot to

Additionally, the box plot has the `freq` parameter (default 15), which controls the number of records included in each individual box on the plot.

## Custom Metrics

In the case that you might want to track some metric that is not provided by default, you can use custom metric functions to get the job done.

Each custom metric function must take 2 arguments, even if they are unused: a `worker` object and a pynvml device `handle` (or handles).

An example custom metric function that tracks power usage across GPUs in a cluster might look like this:

```python
from dask_metrics import custom_metric

@custom_metric('power')
def power_usage(worker, handle):
    return pynvml.nvmlDeviceGetPowerUsage(handle)
```

You use the `custom_metric` decorator to indicate the name of this metric (what will show up as the column name) and whether this metric is run for all devices on a worker separately or all together at once (by setting the kwarg `per_device` in the decorator to `False`).

The previous example is a case of the former and `handle` respresents a single device handle the function is run for. In the case of the latter, `handle` is actually passed a list of the pynvml device handles instead and allows you to make comparisons between all the values for each GPU at a point in time.

Note that custom metric functions are also passed a reference to the `Worker` object representing each worker, giving you access to track information about task states and and anything else Dask is up to.

To let your monitor know to use this custom metric function, make sure to set the `custom_metrics` argument in your call to `Monitor.attach`:

```python
monitor.attach(
    tracking=['mem-util', 'compute-util'],
    custom_metrics=[power_usage]
)
```

## GPU-BDB

1. Install dask-metrics on all nodes of the cluster
2. Start the cluster and connect all the workers
3. Connect the monitor and start recording
4. Run the benchmark
5. Stop the monitor and export metrics to csv

Refer to [How to Use](#how-to-use) for help with starting and stopping the monitor as well as exporting the metrics once the benchmark is complete.

## Additional Information

Currently, in the case that a task submits another task to the cluster during its execution, that task will be considered part of the first when counting job and dag numbers.

Please do not hesitate to report any bugs you might find or suggest useful features.
