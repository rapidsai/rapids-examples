# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:10:44 2021
"""

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


class SimpleTimer:
    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter_ns()
        self.elapsed = self.end - self.start


class ResultsLogger(object):
    def __init__(self, path):
        self.path = path
        self.run_logs = []

    def log(self, row: dict):
        self.run_logs.append(row)

    def write(self):
        df = pd.DataFrame(self.run_logs)
        df.to_csv(self.path)


def scale_workers(client, n_workers, timeout=300):
    client.cluster.scale(n_workers)
    m = len(client.has_what().keys())
    start = end = time.perf_counter_ns()
    while ((m != n_workers) and (((end - start) / 1e9) < timeout)):
        time.sleep(5)
        m = len(client.has_what().keys())
        end = time.perf_counter_ns()
    if (((end - start) / 1e9) >= timeout):
        raise RuntimeError(f"Failed to rescale cluster in {timeout} sec."
                           "Try increasing timeout for very large containers,"
                           "and verify available compute resources.")


def visualize_data_cuml(path, size=(12, 8)):
    # Returns the latencies from the cuML
    # Plots the graph with melted dataframe
    perf_df = pd.read_csv(path)
    perf_df = perf_df.loc[:, ~perf_df.columns.str.contains('^Unnamed:')]
    dd = pd.melt(perf_df,
                 id_vars=['n_workers'],
                 value_vars=['overall', 'data_read', 'data_preprocessing',
                             'hashing_vectorizer', 'tfidf_transformer'],
                 var_name='latency')
    plt.figure(figsize=size, dpi=100, facecolor='w', edgecolor='k')
    sns.boxplot(x='latency', y='value', data=dd, orient="v", hue="n_workers")
    plt.xlabel("Overall Latency and latencies of different stages")
    plt.ylabel("Latency in Seconds")
    plt.show()
    return perf_df, dd


def visualize_data(path, size=(12, 8)):
    # Returns the latencies from the Spark and Scikit results
    # Plots the graph with melted dataframe
    perf_df = pd.read_csv(path)
    perf_df = perf_df.loc[:, ~perf_df.columns.str.contains('^Unnamed:')]
    dd = pd.melt(perf_df,
                 id_vars=['n_workers'],
                 value_vars=['overall', 'data_read', 'data_preprocessing',
                             'hashing_vectorizer', 'tfidf_transformer'],
                 var_name='latency')
    plt.figure(figsize=size, dpi=100, facecolor='w', edgecolor='k')
    sns.boxplot(x='latency', y='value', data=dd, orient="v")
    plt.xlabel("Overall Latency and latencies of different stages")
    plt.ylabel("Latency in Seconds")
    plt.show()
    return perf_df, dd


def visualize_data_spark_adjusted(path):
    # Returns the adjusted dataframe with the latencies of each
    # stage calculated from the cumulative latencies
    perf_df = pd.read_csv(path)
    perf_df = perf_df.loc[:, ~perf_df.columns.str.contains('^Unnamed:')]
    perf_df["tfidf_transformer"] = perf_df["tfidf_transformer"] - \
        perf_df["hashing_vectorizer"]
    perf_df["hashing_vectorizer"] = perf_df["hashing_vectorizer"] - \
        perf_df["data_preprocessing"]
    perf_df["data_preprocessing"] = perf_df["data_preprocessing"] - \
        perf_df["data_read"]
    plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
    dd = pd.melt(perf_df,
                 id_vars=['n_workers'],
                 value_vars=['overall', 'data_read', 'data_preprocessing',
                             'hashing_vectorizer', 'tfidf_transformer'],
                 var_name='latency')
    sns.boxplot(x='latency', y='value', data=dd, orient="v")
    plt.xlabel("Overall Latency and latencies of different stages")
    plt.ylabel("Latency in Seconds")
    plt.show()
    return perf_df, dd
