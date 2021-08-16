import cudf
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "Ensure matplotlib is installed in this environment before using visualization tools"
    )


def _get_data(worker_file, job, metrics):
    df = cudf.read_csv(worker_file)
    section = df[df["job"] == job]

    sections = []
    for metric in metrics:
        if metric not in section:
            raise ValueError(
                f"The requested metric '{metric}' was not found in the provided file"
            )

        m = section[metric].str.split(pat=", ")
        m = m.to_arrow().to_pylist()
        m = cudf.DataFrame(m, dtype="int64")
        m.insert(0, "timestamp", df["timestamp"])
        m._metric_name = metric
        sections.append(m)

    return sections


def lines(worker_file, job, metrics, width=20, height=10, save=None):
    """
    Draws a line chart with overlapping lines for each worker.

    Parameters:
    worker_file : str
        Path to the output csv file representing a worker
    job : int
        Job in run to plot
    metrics : list
        The list of metric names to plot
    freq : int, default 15
        The number of records included in each individual box on the plot
    width : int, default 20
        Plot width in inches
    height : int, default 10
        Plot height in inches
    save : str, optional
        Saves an image of the plot at the specified path if set
    """
    sections = _get_data(worker_file, job, metrics)

    plt.close("all")
    fig, axes = plt.subplots(nrows=len(sections), sharex=True)

    i = 0
    for section in sections:
        try:
            axis = axes[i]
        except:
            axis = axes

        plot = section.to_pandas().plot(
            ax=axis,
            title=section._metric_name,
            x="timestamp",
            y=range(len(section.columns) - 1),
            figsize=(width, height),
            alpha=0.5,
        )
        i += 1

    if save is not None:
        plt.savefig(save, bbox_inches="tight")


def boxes(worker_file, job, metrics, freq=5 * 3, width=20, height=10, save=None):
    """
    Draws a box and whisker plot.

    Parameters:
    worker_file : str
        Path to the output csv file representing a worker
    job : int
        Job in run to plot
    metrics : list
        The list of metric names to plot
    freq : int, default 15
        The number of records included in each individual box on the plot
    width : int, default 20
        Plot width in inches
    height : int, default 10
        Plot height in inches
    save : str, optional
        Saves an image of the plot at the specified path if set
    """
    sections = _get_data(worker_file, job, metrics)

    def process_data(data):
        timestamp = data["timestamp"][::freq].round(2)
        values = (
            data.melt(id_vars=["timestamp"], value_name="val")
            .sort_values(by=["timestamp"])
            .drop(["variable", "timestamp"], 1)
        )

        def chunk(seq, size):
            return (seq[pos : pos + size] for pos in range(0, len(seq), size))

        df = cudf.DataFrame()
        i = 0
        chunk_size = (len(data.columns) - 1) * freq

        for df_chunk in chunk(values, chunk_size):
            df_chunk = df_chunk.reset_index()
            if len(df_chunk.index) < chunk_size:
                for j in range(chunk_size - len(df_chunk.index)):
                    df_chunk.append([np.nan])
            df[timestamp.iloc[i]] = df_chunk["val"]
            i += 1

        return df

    plt.close("all")
    fig, axes = plt.subplots(nrows=len(sections), sharex=True, figsize=(width, height))

    i = 0
    for section in sections:
        try:
            axis = axes[i]
        except:
            axis = axes

        # plot content
        axis.set_title(section._metric_name)
        section = process_data(section).to_pandas()
        bp = axis.boxplot(section)
        plt.xticks(
            list(range(1, len(section.columns) + 1)), section.columns, rotation=45
        )

        # formatting
        colors = ["#aec6cf", "#cedde2", "#cfb7ae"]

        for box in bp["boxes"]:
            box.set(color=colors[0])

        for whisker in bp["whiskers"]:
            whisker.set(color=colors[1], linewidth=3, linestyle=":")

        for cap in bp["caps"]:
            cap.set(color=colors[2], linewidth=2)

        for median in bp["medians"]:
            median.set(color=colors[2], linewidth=2)

        for flier in bp["fliers"]:
            flier.set(marker="D", color=colors[0], alpha=0.5)

        i += 1

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
