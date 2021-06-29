from distributed.diagnostics.plugin import SchedulerPlugin, WorkerPlugin
from dask.distributed import Client, Scheduler
from enum import Enum
from threading import Event
from pathlib import Path
import os
import cudf
import pandas
import asyncio
import time
import pynvml


class Monitor():
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
        
    def attach(self, tracking=[], **kwargs):
        """
        Attaches plugin to scheduler and workers.
        
        Only call once. Calling multiple times will attach
        duplicate plugins, which is generally bad.
        
        Parameters:
        tracking : list[str], optional
            Sets the list of metrics to be tracked on each worker
            > See the list below of metric options
        polling_interval : int, default 200
            The period, in milliseconds, for the monitor to wait
            between polling the GPU
        mem_limit: int, optional
            If set, workers will write collected metrics to disk
            every mem_limit number of readings to free up memory.
            If not set, workers will hold everything in memory
            
        Metrics tracked:
        total-mem: int, int, int, ...
            Total memory in use at a particular point in time, in bytes,
            for each device used by worker
        mem-util : int, int, int, ...
            Percentage of available memory utilized at a particular
            point in time for each device used by worker
        compute-util : int, int, int, ...
            Percentage of available compute capability utilized at a
            particular point in time for each device used by worker
        """
        self.monitor = SchedulerMonitor(self.client, tracking=tracking, **kwargs)
        
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
            self.send({
                'op': 'update_tracking',
                'tracking': tracking
            })
            
            # new metrics might be incompatible format with current
            # metrics, so the old ones must be cleared
            self.metrics = []
            self.metric_state = MetricState.NONE
        self.send({'op': 'start_recording'}) # start collecting metrics
        
    def stop(self):
        """
        Stops the collection of metrics on all workers.
        """
        self.send({'op': 'stop_recording'})
        
    def request_metrics(self):
        """
        Pulls all metrics collected until now by the workers and stores them in memory.
        """
        if self.metric_state == MetricState.HAS_ALL:
            return
        
        def func(data, all_metrics): # callback func to recieve metrics
            self.metrics = data
            self.metric_state = MetricState.HAS_ALL if all_metrics else MetricState.HAS_SOME
            self.event.set()
        
        self.client._stream_handlers.update({'give_metrics': func})
        self.send({'op': 'send_metrics'})
        self.event.wait()
        self.event.clear()
    
    def clear_metrics(self, **kwargs):
        """
        Clears the stored metrics on workers and scheduler.
        """
        self.metrics = []
        self.metric_state = MetricState.NONE
        self.send({'op': 'clear_metrics'})
        
    def to_csv(self, path, fname='metrics'):
        """
        Exports currently stored metrics to designated csv files.
        
        Pulls metrics up to the scheduler from each worker first if not done already.
        
        Parameters:
        path : str
            Path to the folder where metrics files are exported
        fname : str, default "metrics"
            File name prefix for each worker csv
            Ex: metrics_0.csv, metrics_1.csv, ...
        """
        self.request_metrics()
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
            if 'mem-util' not in df.columns:
                raise KeyError("Worker needs 'mem-util' tracked to compute peak memory utilization")
            
            # magically expand list column into dataframe
            col = df['mem-util'].str.split(pat=', ')
            col = col.to_arrow().to_pylist()
            devices = cudf.DataFrame(col)
            
            return [int(devices[d].max()) for d in devices.columns]
        
        if workers and len(workers) == 1: # single worker
            return compute_peak(workers[0])
        else: # all workers
            peaks = []
            wrk = workers if workers else range(len(self.metrics))
            for i in wrk:
                peaks.append(compute_peak(i))
            return peaks
        

class SchedulerMonitor(SchedulerPlugin):
    def __init__(self, client, polling_interval=200, tracking=[], mem_limit=None):
        self.polling_interval = polling_interval/1000.0 # ms -> sec
        self.tracking_list = tracking
        self.mem_limit = mem_limit if (mem_limit and mem_limit > 0) else 0
        self.running = False
        self.start = 0
        self.stop = 0
        self.job_number = 0
        self.dag_number = 0
        self.dag_in_job = 0
        self.jobs = []
        self.dags = []
        self.metrics = []
        self.metric_state = MetricState.NONE
        
        self.clients_connected = {}
        
        self.attach(client)
    
    def attach(self, client):
        # adds the plugin to the scheduler of a cluster
        self.scheduler = client.cluster.scheduler
        self.scheduler.stream_handlers.update({
            'start_recording': self.start_recording,
            'stop_recording': self.stop_recording,
            'update_tracking': self.update_tracking,
            'report_metrics': self.recieve_metrics,
            'clear_metrics': self.clear_metrics,
            'aggregate_metrics': lambda *args, **kwargs: self.broadcast({'op': 'send_metrics'}),
            'send_metrics': self.send_to_client
        })
        
        self.scheduler.add_plugin(self)
        options = [
            self.polling_interval,
            self.tracking_list,
            self.mem_limit
        ]
        client.register_worker_plugin(WorkerMonitor(*options))
    
    def broadcast(self, msg, client=None):
        # send message (usually a dict) to all workers
        for addr, comm in self.scheduler.stream_comms.items():
            comm.send(msg)
            
    def add_client(self, scheduler=None, client=None, **kwargs):
        if not self.worker_client(client):
            self.clients_connected[client] = len(self.jobs)
            self.jobs.append(0)
              
    def remove_client(self, scheduler=None, client=None, **kwargs):
        if client in self.clients_connected:
            del self.clients_connected[client]
    
    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None, **kwargs):
        # runs every time a new dag is submitted to the cluster
        if not self.worker_client(kwargs['client']):
            self.jobs[self.clients_connected[kwargs['client']]] += 1
            for key in keys:
                self.dags.append(key)
        
    def transition(self, key, start, finish, **kwargs):
        # runs every time a task changes state   
        if start == 'processing' and finish == 'memory':
            if key in self.dags:
                self.dag_number += 1
                self.dag_in_job += 1
                self.broadcast({
                    'op': 'dag_num',
                    'val': self.dag_in_job
                })
                
                if self.dag_in_job == self.jobs[self.job_number]:
                    self.dag_in_job = 0 # reset dag num
                    self.job_number += 1
                    self.broadcast({
                        'op': 'job_num',
                        'val': self.job_number
                    })
    
    def worker_client(self, client):
        if 'Client-worker-' in client:
            return True
        return False
    
    def start_recording(self, **kwargs):
        self.broadcast({'op': 'start_recording'})
        self.thread_lock = asyncio.Lock()
        self.start = time.time()
        self.stop = 0
        self.running = True
    
    def stop_recording(self, **kwargs):
        self.broadcast({'op': 'stop_recording'})
        self.stop = time.time()
        self.running = False
        
    def update_tracking(self, tracking, **kwargs):
        # updates list of tracked metrics
        self.tracking_list = tracking
        self.broadcast({
            'op': 'update_tracking',
            'tracking': tracking
        })
        self.clear_metrics()
    
    async def pull_metrics(self):
        async with self.thread_lock:
            if self.metric_state == MetricState.HAS_ALL:
                return # no new metrics to collect
            
            self.metric_state = MetricState.GATHERING
            self.metrics = []
            self.event = asyncio.Event()
            self.workers_not_reported = len(self.scheduler.stream_comms)
            
            self.broadcast({'op': 'send_metrics'}) # ask workers for metrics
            await self.event.wait() # wait for them all to get back
            self.metric_state = MetricState.HAS_SOME if self.running else MetricState.HAS_ALL
            
    def recieve_metrics(self, metrics, **kwargs):
        # runs when a single worker sends a metrics report
        self.metrics.append(metrics)
        self.workers_not_reported -= 1
        if self.workers_not_reported == 0:
            self.event.set()
        
    async def send_to_client(self, client):
        # sends metrics back to a specific client
        await self.pull_metrics()
        c = self.scheduler.client_comms.get(client)
        c.send({
            'op': 'give_metrics',
            'data': self.metrics,
            'all_metrics': self.metric_state == MetricState.HAS_ALL
        })
    
    def clear_metrics(self, **kwargs):
        self.job_number = 0
        self.dag_number = 0
        self.metric_state = MetricState.NONE
        self.metrics = [] # clear on scheduler
        self.broadcast({'op': 'clear_metrics'}) # clear on workers

        
class WorkerMonitor(WorkerPlugin):
    def __init__(self, interval, tracking, limit):
        # runs once before given to workers
        self.polling_interval = interval
        self.update_tracking(tracking)
        self.mem_limit = limit
        self.dump_loc = 'temp'
    
    def setup(self, worker):
        # runs to initialize on each worker
        self.worker = worker
        self.start = 0
        self.stop = 0
        self.job_number = 0
        self.dag_number = 0
        
        worker.stream_handlers.update({
            'start_recording': self.start_recording,
            'stop_recording': self.stop_recording,
            'clear_metrics': self.clear_metrics,
            'job_num': self.update_job_num,
            'dag_num': self.update_dag_num,
            'send_metrics': self.report_metrics
        })
        
    def start_recording(self):
        self.start = time.time()
        self.stop = 0
        pynvml.nvmlInit()
        
        # start loop
        self.loop = asyncio.ensure_future(self.log_metrics())
        
    def stop_recording(self):
        self.stop = time.time()
        if not self.loop.cancelled():
            self.loop.cancel() # shut down the loop
        pynvml.nvmlShutdown()
    
    def update_job_num(self, val):
        self.job_number = val
    
    def update_dag_num(self, val):
        self.dag_number = val
        
    def report_metrics(self):
        # send metrics back to scheduler
        # TODO: read back file dumps to send in chunks
        bs = self.worker.batched_stream
        bs.send({
            'op': 'report_metrics',
            'metrics': self.metrics
        })
    
    def clear_metrics(self, **kwargs):
        # clears all stored metrics
        self.job_number = 0
        self.dag_number = 0
        self.metrics = {
            'job': [],
            'dag' : [],
            'timestamp': []
        }
        self.metrics.update({x: [] for x in self.tracking_list})
        # TODO: clear temporary metrics on disk
        
    def update_tracking(self, tracking, **kwargs):
        # updates list of tracked metrics
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
            'total-mem': total_mem,
            'mem-util': mem_util,
            'compute-util': compute_util
        }
        self.tracking_list = {k: v for k, v in operations.items() if k in tracking}
        self.clear_metrics()
        
    def dump_partial(self):
        # writes metrics in memory to disk temporarily
        Path(self.dump_loc).mkdir(parents=True, exist_ok=True)
        path = f"{self.dump_loc}/{self.worker.id}.csv"
        df = pandas.DataFrame(self.metrics)
        
        if Path(path).is_file(): # append
            with open(path, 'a') as f:
                df.to_csv(f, header=False)
        else: # create
            df.to_csv(path)
        self.clear_metrics()
    
    async def log_metrics(self):
        # the loop that polls the gpu for metrics using pynvml
        while self.stop == 0:
            # universal metrics always tracked
            self.metrics['job'].append(self.job_number)
            self.metrics['dag'].append(self.dag_number)
            self.metrics['timestamp'].append(time.time() - self.start)

            # collect extra metrics defined in the tracking list
            for name, op in self.tracking_list.items():
                self.metrics[name].append(op())

            # dump metrics to disk if enough collected in memory
            if self.mem_limit > 0 and len(self.metrics) >= self.mem_limit:
                self.dump_partial()
            await asyncio.sleep(self.polling_interval) # wait for next cycle
    
    def device_info(self, operation):
        # applies operation over all device handles and returns
        # a comma separated string listing the results
        handle = pynvml.nvmlDeviceGetHandleByIndex # fn alias
        device_count = pynvml.nvmlDeviceGetCount()
        results = [str(operation(handle(i))) for i in range(device_count)]
        return ", ".join(results) # join together with commas


class MetricState(Enum):
    # Keeps track of what metrics something has
    # or if it is in the process of gathering them
    NONE = 0
    GATHERING = 1
    HAS_SOME = 2
    HAS_ALL = 3