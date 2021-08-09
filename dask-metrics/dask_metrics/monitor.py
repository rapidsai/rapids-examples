from distributed.diagnostics.plugin import SchedulerPlugin, WorkerPlugin
from dask.distributed import Client, Scheduler
from distributed.protocol.pickle import dumps, loads
from distributed.core import connect
from enum import Enum
from threading import Event
from pathlib import Path
import pandas as pd
import os
import cudf
import asyncio
import time
import pynvml


class Monitor:
    def __init__(self, client):
        """
        Attaches a monitor to all the workers in the cluster the client is
        connected to when started. Each monitor tracks metrics about the gpus
        the worker is using and stores them locally.
        
        Start and stop metric collection on all workers using the start() and
        stop() functions respectively.
        
        Metrics can be pulled the the client using request_metrics() or dumped
        into a csv using to_csv().
        """
        self.client = client
        self.metric_state = MetricState.NONE
        self.metrics = []
        self.monitor = None
        self.event = Event()
        self.send = self.client._send_to_scheduler

    def attach(self, job_type="client", **kwargs):
        """
        Attaches plugin to scheduler and workers.
        
        Only call once. Calling multiple times will attach
        duplicate plugins, which is generally bad.
        
        Parameters:
        tracking : list[str], optional
            Sets the list of metrics to be tracked on each worker
            > See the list below of metric options
        job_type : str, default 'client'
            The method by which job number is tracked. 'client' will
            link job number to client connections, while 'manual'
            will require calls to new_job() to increment job number.
        polling_interval : int, default 200
            The period, in milliseconds, for the monitor to wait
            between polling the GPU
        mem_limit : int, optional
            If set, workers will write collected metrics to disk
            every mem_limit number of readings to free up memory.
            If not set, workers will hold everything in memory
        dump_loc : str, default 'temp'
            If mem_limit > 0, dump_loc specifies the directory
            on disk for each worker where temporary metrics
            are stored
            
        Metrics tracked:
        total-mem : int, int, int, ...
            Total memory in use at a particular point in time, in bytes,
            for each device used by worker
        mem-util : int, int, int, ...
            Percentage of available memory utilized at a particular
            point in time for each device used by worker
        compute-util : int, int, int, ...
            Percentage of available compute capability utilized at a
            particular point in time for each device used by worker
        """
        self.job_type = job_type
        self.monitor = SchedulerMonitor(job_type)
        self._register_scheduler_plugin(self.monitor)
        self.client.register_worker_plugin(WorkerMonitor(**kwargs))
        # self.client.register_worker_plugin(WorkerPlugin())

    def start(self, tracking=[]):
        """
        Starts the collection of metrics on all workers.
        
        Parameters:
        tracking : list[str], optional
            Sets the list of metrics to be tracked on each worker. Clears
            stored metrics if the list is being updated
            > See attach() for metric options
        """
        if len(tracking) > 0:
            # update tracking list if new one passed
            self.send({"op": "update_tracking", "tracking": tracking})

            # new metrics might be incompatible format with current
            # metrics, so the old ones must be cleared
            self.metrics = []
            self.metric_state = MetricState.NONE
        if len(self.metrics) > 0:
            self.metric_state = MetricState.HAS_SOME
        self.send({"op": "start_recording"})  # start collecting metrics

    def stop(self, force=False):
        """
        Stops the collection of metrics on all workers.
        """
        self.send({"op": "stop_recording", "force": force})

    def shutdown(self):
        """
        Removes the monitor from the cluster and deletes any remaining
        temporary files that might be left on workers.
        """
        self.send({"op": "metrics_shutdown"})

    def new_job(self, job=None):
        """
        Used to indicate which job is currently executing to the monitor. Only used
        if job_type is set to 'manual' upon monitor startup.
        
        Parameters:
        job : int, optional
            If set, the new job number is set to the value passed to job,
            otherwise the job number is incremented by one.
        """
        if self.job_type == "manual":
            self.send({"op": "new_job", "job": job})
        else:
            print("Manual job signalling not enabled")

    def request_metrics(self):
        """
        Pulls all metrics collected until now by the workers and stores them in memory.
        """
        if self.metric_state == MetricState.HAS_ALL:
            return

        def func(data, all_metrics, successful):  # callback func to recieve metrics
            self.metrics = data
            self.metric_state = (
                MetricState.HAS_ALL if all_metrics else MetricState.HAS_SOME
            )
            self.successful_jobs = successful
            self.event.set()

        self.client._stream_handlers.update({"give_metrics": func})
        self.send({"op": "send_metrics"})
        self.event.wait()
        self.event.clear()

    def clear_metrics(self, **kwargs):
        """
        Clears the stored metrics on workers and scheduler.
        """
        self.metrics = []
        self.metric_state = MetricState.NONE
        self.send({"op": "clear_metrics"})

    def to_csv(self, path, fname="worker", clear_dir=False):
        """
        Exports currently stored metrics to designated csv files.
        
        Pulls metrics up to the scheduler from each worker first if not done already.
        
        Parameters:
        path : str
            Path to the folder where metrics files are exported
        fname : str, default "metrics"
            File name prefix for each worker csv
            Ex: metrics_0.csv, metrics_1.csv, ...
        clear_dir : bool, default False
            Clears existing files out of the target directory
            before writing new ones if true
        """
        self.request_metrics()

        if clear_dir:
            for f in os.listdir(path):
                p = os.path.join(path, f)
                if not os.path.isdir(p):
                    os.remove(p)

        for idx, worker in enumerate(self.metrics):
            # creates file for each worker
            df = cudf.DataFrame(worker)
            df.to_csv(f"{path}{os.path.sep}{fname}_{idx}.csv", index=False)

    def peak_memory(self, workers=None):
        """
        Gets the peak memory utilization for each worker.
        
        Pulls metrics up to the scheduler from each worker first if not done already.
        
        Parameters:
        workers : list, optional
            If workers is specified, peak_memory will only return the peak memory utilization
            for those workers. Otherwise, it will return all of them
        """
        self.request_metrics()

        def compute_peak(idx):
            # computes peak utilization for single worker
            df = cudf.DataFrame(self.metrics[idx])
            if "mem-util" not in df.columns:
                raise KeyError(
                    "Worker needs 'mem-util' tracked to compute peak memory utilization"
                )

            # magically expand list column into dataframe
            col = df["mem-util"].str.split(pat=", ")
            col = col.to_arrow().to_pylist()
            devices = cudf.DataFrame(col)

            return [int(devices[d].max()) for d in devices.columns]

        if workers and len(workers) == 1:  # single worker
            return compute_peak(workers[0])
        else:  # all workers
            peaks = []
            wrk = workers if workers else range(len(self.metrics))
            for i in wrk:
                peaks.append(compute_peak(i))
            return peaks

    def successful_jobs(self):
        """
        Returns an list of lists, where each list is a job and each list
        item is a boolean value representing whether that dag finished
        successfully or encountered an error during its execution.
        """
        self.request_metrics()
        return self.successful_jobs

    def _register_scheduler_plugin(self, plugin):
        ## Workaround until Client.register_scheduler_plugin()
        ## makes its way into a stable release
        async def coro():
            try:  # create connection to scheduler
                comm = await connect(
                    self.client.scheduler.address,
                    timeout=None,
                    **self.client.connection_args,
                )
                comm.name = "Client->Scheduler"
            except Exception:
                print("failed to connect")

            p = dumps(plugin, protocol=4)  # serialize plugin

            def func(dask_scheduler, plugin=None):
                p = loads(plugin)  # deserialize
                dask_scheduler.add_plugin(p)
                p.register_handlers(dask_scheduler)

            await comm.write(
                {
                    "op": "run_function",
                    "function": dumps(func),
                    "kwargs": dumps({"plugin": p}),
                }
            )
            comm.close()

        loop = asyncio.get_event_loop()
        task = loop.create_task(coro())
        loop.run_until_complete(task)


class SchedulerMonitor(SchedulerPlugin):
    def __init__(self, job_type):
        self.running = False
        self.start = 0
        self.stop = 0

        self.job_number = 0
        self.dag_number = 0
        self.dag_in_job = 0

        self.jobs = []
        self.dags = []
        self.successful_jobs = []

        self.metrics = []
        self.metric_state = MetricState.NONE
        self.clients_connected = {}
        self.jobs_done = None

        if job_type in ("client", "manual"):
            self.job_type = job_type
        else:
            raise ValueError(
                f"Job type must be either 'client' or 'manual'. Given: '{job_type}'"
            )

    def register_handlers(self, scheduler):
        self.scheduler = scheduler
        scheduler.stream_handlers.update(
            {
                "start_recording": self.start_recording,
                "stop_recording": self.stop_recording,
                "new_job": self.new_job,
                "update_tracking": self.update_tracking,
                "report_metrics": self.recieve_metrics,
                "clear_metrics": self.clear_metrics,
                "aggregate_metrics": lambda *args, **kwargs: self.broadcast(
                    {"op": "send_metrics"}
                ),
                "send_metrics": self.send_to_client,
                "metrics_shutdown": self.shutdown,
            }
        )

    def broadcast(self, msg, client=None):
        ## send message (usually a dict) to all workers
        for addr, comm in self.scheduler.stream_comms.items():
            comm.send(msg)

    def add_client(self, scheduler=None, client=None, **kwargs):
        # check for client-based job tracking logic
        if not self.worker_client(client) and self.job_type == "client":
            self.clients_connected[client] = len(self.jobs)

            # increment job number if all dags in last job complete
            if (
                self.job_number == len(self.jobs) - 1
                and self.dag_in_job == self.jobs[self.job_number] - 1
                and len(self.jobs) != 0
            ):
                self.job_number += 1
                self.dag_in_job = 0
                self.broadcast(
                    {"op": "job_state", "job": self.job_number, "dag": self.dag_in_job}
                )
            self.jobs.append(0)
            self.successful_jobs.append([])

    def remove_client(self, scheduler=None, client=None, **kwargs):
        if client in self.clients_connected:
            del self.clients_connected[client]

    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None, **kwargs):
        ## runs every time a new dag is submitted to the cluster
        if not self.worker_client(kwargs["client"]):
            # check for client-based job tracking logic
            if self.job_type == "client":
                if kwargs["client"] not in self.clients_connected:
                    self.add_client(client=kwargs["client"])
                client = self.clients_connected[kwargs["client"]]

                # check if all dags in all jobs complete
                if (
                    self.job_number == len(self.jobs) - 1
                    and self.dag_in_job == self.jobs[self.job_number] - 1
                ):
                    # must be new submission for last job, increment dag
                    self.dag_number += 1
                    self.dag_in_job += 1
                    self.broadcast({"op": "job_state", "dag": self.dag_in_job})

                self.jobs[client] += 1
                if self.jobs_done.is_set():
                    self.jobs_done.clear()

            # add dag key to list
            for key in keys:
                self.dags.append(key)

    def transition(self, key, start, finish, **kwargs):
        ## runs every time a task changes state
        if start == "processing" and finish == ("memory" or "error"):
            if key in self.dags:
                # check for client-based job tracking logic
                if self.job_type == "client":
                    # check if all dags complete for job
                    if self.dag_in_job == self.jobs[self.job_number] - 1:
                        if self.job_number < len(self.jobs) - 1:
                            # if there is another job waiting, start it
                            self.dag_in_job = 0
                            self.job_number += 1
                        else:
                            # otherwise there is no work to be done
                            if (
                                not self.jobs_done.is_set()
                                and len(self.clients_connected) == 0
                            ):
                                self.jobs_done.set()
                    else:
                        # job not complete, keep counting dags
                        self.dag_number += 1
                        self.dag_in_job += 1
                else:
                    # manual job counting, just keep incrementing dag number
                    self.dag_number += 1
                    self.dag_in_job += 1

                # notify workers of job state
                self.broadcast(
                    {"op": "job_state", "job": self.job_number, "dag": self.dag_in_job}
                )
            self.successful_jobs[self.job_number].append(finish == "memory")

    def worker_client(self, client):
        return "Client-worker-" in client

    def start_recording(self, **kwargs):
        self.broadcast({"op": "start_recording"})
        self.thread_lock = asyncio.Lock()
        self.start = time.time()
        self.stop = 0
        self.running = True
        if len(self.metrics) > 0:
            self.metric_state = MetricState.HAS_SOME
        self.jobs_done = asyncio.Event()

    async def stop_recording(self, force=False, **kwargs):
        if not force:
            await self.jobs_done.wait()
        self.broadcast({"op": "stop_recording"})
        self.stop = time.time()
        self.running = False

    def shutdown(self, **kwargs):
        self.broadcast({"op": "metrics_shutdown"})
        self.stop = time.time()
        self.running = False
        self.scheduler.remove_plugin(self)

    def new_job(self, job=None, **kwargs):
        if job is not None:
            self.job_number = job
        else:
            self.job_number += 1
        self.dag_in_job = 0
        self.broadcast(
            {"op": "job_state", "job": self.job_number, "dag": self.dag_in_job}
        )

    def update_tracking(self, tracking, **kwargs):
        ## updates list of tracked metrics
        self.tracking_list = tracking
        self.broadcast({"op": "update_tracking", "tracking": tracking})
        self.clear_metrics()

    async def pull_metrics(self):
        async with self.thread_lock:
            if self.metric_state == MetricState.HAS_ALL:
                return  # no new metrics to collect

            self.metric_state = MetricState.GATHERING
            self.metrics = []
            self.event = asyncio.Event()
            self.workers_not_reported = len(self.scheduler.stream_comms)

            self.broadcast({"op": "send_metrics"})  # ask workers for metrics
            await self.event.wait()  # wait for them all to get back
            self.metric_state = (
                MetricState.HAS_SOME if self.running else MetricState.HAS_ALL
            )

    def recieve_metrics(self, metrics, **kwargs):
        ## runs when a single worker sends a metrics report
        self.metrics.append(metrics)
        self.workers_not_reported -= 1
        if self.workers_not_reported == 0:
            self.event.set()

    async def send_to_client(self, client):
        ## sends metrics back to a specific client
        await self.pull_metrics()
        c = self.scheduler.client_comms.get(client)
        c.send(
            {
                "op": "give_metrics",
                "data": self.metrics,
                "all_metrics": self.metric_state == MetricState.HAS_ALL,
                "successful": self.successful_jobs,
            }
        )

    def clear_metrics(self, **kwargs):
        self.job_number = 0
        self.dag_number = 0
        self.dag_in_job = 0
        self.metric_state = MetricState.NONE
        self.metrics = []  # clear on scheduler
        self.broadcast({"op": "clear_metrics"})  # clear on workers


class WorkerMonitor(WorkerPlugin):
    def __init__(self, tracking=[], polling_interval=200, mem_limit=0, dump_loc="temp"):
        ## runs once before given to workers
        self.metrics_on_disk = False
        self.update_tracking(tracking)
        self.polling_interval = polling_interval / 1000.0  # ms -> sec
        self.mem_limit = mem_limit
        self.dump_loc = dump_loc

    def setup(self, worker):
        ## runs to initialize on each worker
        self.worker = worker
        self.start = 0
        self.stop = 0
        self.job_number = 0
        self.dag_number = 0

        worker.stream_handlers.update(
            {
                "start_recording": self.start_recording,
                "stop_recording": self.stop_recording,
                "clear_metrics": self.clear_metrics,
                "job_state": self.update_job_state,
                "send_metrics": self.report_metrics,
                "metrics_shutdown": self.shutdown,
            }
        )

    def start_recording(self):
        self.start = time.time()
        self.stop = 0
        pynvml.nvmlInit()

        # start loop
        self.loop = asyncio.ensure_future(self.log_metrics())

    def stop_recording(self):
        self.stop = time.time()
        if not self.loop.cancelled():
            self.loop.cancel()  # shut down the loop
        pynvml.nvmlShutdown()

    async def shutdown(self):
        self.stop_recording()
        self.clear_metrics(clear_disk=True)
        await self.worker.plugin_remove(self)

    def update_job_state(self, job=None, dag=None):
        if job is not None:
            self.job_number = job
        if dag is not None:
            self.dag_number = dag

    @property
    def disk_location(self):
        ## location of dumped metrics on disk
        return f"{self.dump_loc}/{self.worker.id}.csv"

    def report_metrics(self):
        ## send metrics back to scheduler
        if self.metrics_on_disk:  # read back file dumps
            report = pd.read_csv(self.disk_location)
            report.append(pd.DataFrame(self.metrics))
            report = report.to_dict(orient="list")
        else:
            report = self.metrics

        # remove weird to_dict side effect
        if "Unnamed: 0" in report:
            del report["Unnamed: 0"]

        # send to scheduler
        bs = self.worker.batched_stream
        bs.send({"op": "report_metrics", "metrics": report})

    def clear_metrics(self, clear_disk=False, clear_status=True, **kwargs):
        ## clears all stored metrics
        self.metrics = {"job": [], "dag": [], "timestamp": []}
        self.metrics.update({x: [] for x in self.tracking_list})

        # clear temporary metrics on disk
        if clear_disk and self.metrics_on_disk:
            os.remove(self.disk_location)
            self.metrics_on_disk = False
        if clear_status:
            self.job_number = 0
            self.dag_number = 0

    def update_tracking(self, tracking, **kwargs):
        ## updates list of tracked metrics
        def total_mem():
            op = lambda handle: pynvml.nvmlDeviceGetMemoryInfo(handle).used
            return self.device_info(op)

        def mem_util():
            op = lambda handle: pynvml.nvmlDeviceGetUtilizationRates(handle).memory
            return self.device_info(op)

        def compute_util():
            op = lambda handle: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            return self.device_info(op)

        operations = {
            "total-mem": total_mem,
            "mem-util": mem_util,
            "compute-util": compute_util,
        }
        self.tracking_list = {k: v for k, v in operations.items() if k in tracking}
        self.clear_metrics(clear_disk=True)

    def dump_partial(self):
        ## writes metrics in memory to disk temporarily
        # TODO: delete on cluster shutdown ***
        Path(self.dump_loc).mkdir(parents=True, exist_ok=True)
        path = self.disk_location
        df = pd.DataFrame(self.metrics)

        if Path(path).is_file():  # append
            with open(path, "a") as f:
                df.to_csv(f, header=False)
        else:  # create
            df.to_csv(path)

        self.clear_metrics(clear_status=False)
        self.metrics_on_disk = True

    async def log_metrics(self):
        ## the loop that polls the gpu for metrics using pynvml
        while self.stop == 0:
            # universal metrics always tracked
            self.metrics["job"].append(self.job_number)
            self.metrics["dag"].append(self.dag_number)
            self.metrics["timestamp"].append(time.time() - self.start)

            # collect extra metrics defined in the tracking list
            for name, op in self.tracking_list.items():
                self.metrics[name].append(op())

            # dump metrics to disk if enough collected in memory
            if self.mem_limit > 0 and len(self.metrics) >= self.mem_limit:
                self.dump_partial()
            await asyncio.sleep(self.polling_interval)  # wait for next cycle

    def device_info(self, operation):
        ## applies operation over all device handles and returns
        ## a comma separated string listing the results
        handle = pynvml.nvmlDeviceGetHandleByIndex  # fn alias
        device_count = pynvml.nvmlDeviceGetCount()
        results = [str(operation(handle(i))) for i in range(device_count)]
        return ", ".join(results)  # join together with commas


class MetricState(Enum):
    ## Keeps track of what metrics something has
    ## or if it is in the process of gathering them
    NONE = 0
    GATHERING = 1
    HAS_SOME = 2
    HAS_ALL = 3
