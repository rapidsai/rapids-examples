### Benchmarking latency of a NLP tf-idf pipeline across Sklearn, cuML and Spark using Hashing Vectorizer. 

**Note:** All tests are done on a bare-metal server with RAPIDS 21.08a conda environment. The PySpark version we have used is 3.1.2.

For the study, we have used a subset of the [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), which is a dataset of customer reviews of products available on the Amazon platform. We particularly use only the reviews of PC products in this study (no particular reason). While the data is available in both tab separated (`tsv`) and `parquet` file format, we will use the parquet file format here. For the test however, downloading the parquet files in the notebook is extremely slow. Hence we have downloaded the parquet files in the DGX box before running the tests and then read from disk in the notebooks.

The dataset can be downloaded locally by the following (assuming you have `aws-cli` set up):
```
aws s3 cp s3://amazon-reviews-pds/parquet/product_category=Books/ ./data/product_category=Books/ --recursive
```
The size of the compressed parquet files on disk is ~11GB. The total number of rows with reviews is around 21 million (20,725,971). 

We have the following 3 notebooks:
1. [`NLP-pipeline-sklearn.ipynb`](./NLP-pipeline-sklearn.ipynb) is the standard single CPU `sklearn` pipeline. This is a baseline.
2. [`NLP-pipeline-spark.ipynb`](./NLP-pipeline-spark.ipynb) is the multi CPU `spark` pipeline. Here, we allow the Spark cluster to use all the 80 CPU cores of the DGX box. 
3. [`NLP-pipeline-cuML.ipynb`](./NLP-pipeline-cuML.ipynb) is the standard multi GPU `cuML` pipeline leveraging Dask. We do a  strong scaling test using 2, 4, 6 and 8 (32GB V100) GPUs on a single DGX box.

The following are some numbers:

|                      | Overall End to End <br>(without persisting <br>intermediate stages) | Data Read       | Data Preprocessing | Hashing Vectorizer | Tf-idf Transformer   | Sample <br>Runs |
|----------------------|:-------------------------------------------------------------------:|-----------------|--------------------|--------------------|----------------------|-----------------|
| Scikit-Learn         | 4603.981 &#177; 50.242                                                  | 66.078 &#177; 0.724 | 3624.540 &#177; 15.767 | 824.764 &#177; 2.552   | 88.385 &#177; 31.166     | 2               |
| Apache Spark         | 466.508 &#177; 7.330                                                    | 9.393 &#177; 7.77   | 79.220 &#177; 6.842    | 134.934 &#177; 8.527   | 244.402.126 &#177; 9.647 | 5               |
| cuML + Dask (4 GPUs) | 31.170 &#177; 1.190                                                     | 2.817 &#177; 0.025  | 10.416 &#177; 0.613    | 11.033 &#177; 0.346    | 1.987 &#177; 0.075       | 5               |

---

Comparing the end to end latency between Apache Spark and cuML + Dask pipeline, we see a **14.5x** speedup with the latter with just 6 GPUs !! 😎 😎
