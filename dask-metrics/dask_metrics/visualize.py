import cudf
import numpy as np
import matplotlib.pyplot as plt

def _get_data(worker_file, job):
    df = cudf.read_csv(worker_file)
    col = df[df['job'] == job]

    compute = col['compute-util'].str.split(pat=', ')
    compute = compute.to_arrow().to_pylist()
    compute = cudf.DataFrame(compute, dtype='int16')
    compute.insert(0, 'timestamp', df['timestamp'])

    memory = col['mem-util'].str.split(pat=', ')
    memory = memory.to_arrow().to_pylist()
    memory = cudf.DataFrame(memory, dtype='int16')
    memory.insert(0, 'timestamp', df['timestamp'])
    
    return compute, memory

def lines(worker_file, job, width=20, height=10, save=None):
    """
    Draws a line chart with overlapping lines for each worker.
    
    Parameters:
    worker_file : str
        Path to the output csv file representing a worker
    job : int
        Job in run to plot
    freq : int, default 15
        The number of records included in each individual box on the plot
    width : int, default 20
        Plot width in inches
    height : int, default 10
        Plot height in inches
    save : str, optional
        Saves an image of the plot at the specified path if set
    """
    compute, memory = _get_data(worker_file, job)
    
    plt.close('all')
    fig, axes = plt.subplots(nrows=2, sharex=True)
    compute.to_pandas().plot(ax=axes[0], title='Compute Utilization (%)', x='timestamp', y=range(8), figsize=(width, height))
    memory.to_pandas().plot(ax=axes[1], title='Memory Utilization (%)', x='timestamp', y=range(8), figsize=(width, height))
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight')

def boxes(worker_file, job, freq=5*3, width=20, height=10, save=None):
    """
    Draws a box and whisker plot.
    
    Parameters:
    worker_file : str
        Path to the output csv file representing a worker
    job : int
        Job in run to plot
    freq : int, default 15
        The number of records included in each individual box on the plot
    width : int, default 20
        Plot width in inches
    height : int, default 10
        Plot height in inches
    save : str, optional
        Saves an image of the plot at the specified path if set
    """
    compute, memory = _get_data(worker_file, job)
    
    def process_data(data):
        timestamp = data['timestamp'][::freq].round(2)
        values = data.melt(id_vars=['timestamp'], value_name='val').sort_values(by=['timestamp']).drop(['variable', 'timestamp'], 1)
        
        def chunk(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
        df = cudf.DataFrame()
        i = 0
        chunk_size = (len(data.columns) - 1) * freq

        for df_chunk in chunk(values, chunk_size):
            df_chunk = df_chunk.reset_index()
            if len(df_chunk.index) < chunk_size:
                for j in range(chunk_size - len(df_chunk.index)):   
                    df_chunk.append([np.nan])
            df[timestamp.iloc[i]] = df_chunk['val']
            i += 1
        
        return df
    
    plt.close('all')
    fig, axes = plt.subplots(nrows=2, sharex=True)
    
    compute = process_data(compute).to_pandas()
    compute.plot.box(title='Compute Utilization (%)', ax=axes[0], figsize=(width, height), fontsize=8)

    memory = process_data(memory).to_pandas()
    axes[1].boxplot(memory)
    axes[1].set_title('Memory Utilization (%)')
    plt.xticks(list(range(1, len(compute.columns) + 1)), compute.columns, rotation=45)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
