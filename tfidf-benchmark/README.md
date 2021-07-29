### Benchmarking latency of a NLP tf-idf pipeline across Sklearn, cuML and Spark using Hashing Vectorizer. 

**Note:** All tests are done from within a `rapidsai/rapidsai-nightly:21.08-cuda11.2-runtime-ubuntu18.04-py3.8` container. PySpark version is 3.1.2.

For the study, we have used a subset of the [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), which is a dataset of customer reviews of products available on the Amazon platform. We particularly use only the reviews of PC products in this study. The data is available in both tab separated (`tsv`) and `parquet` file format. We will use the parquet file format. For the test however, downloading the parquet files in the notebook is extremely slow. Hence we have downloaded the parquet files in the DGX box before running the tests and then read from disk in the notebooks.

The dataset can be downoaded locally by the following (assuming you have `aws-cli` set up):
```
aws s3 cp s3://amazon-reviews-pds/parquet/product_category=PC/ ./data/product_category=PC/ --recursive
```
The size of the compressed parquet files on disk is ~2.1GB. The total number of rows with reviews is around 7 million (7,004,147). 

We have the following 3 notebooks:
1. [`NLP-pipeline-sklearn.ipynb`](./NLP-pipeline-sklearn.ipynb) is the standard single CPU `sklearn` pipeline. This is a baseline.
2. [`NLP-pipeline-spark.ipynb`](./NLP-pipeline-spark.ipynb) is the multi CPU `spark` pipeline. Here, we allow the Spark cluster to use all the 80 CPU cores of the DGX box and 300GB allocated memory. 
3. [`NLP-pipeline-cuML.ipynb`](./NLP-pipeline-cuML.ipynb) is the standard multi GPU `cuML` pipeline leveraging Dask. We do a  strong scaling test using 2, 4 and 6 (32GB V100) GPUs on a single DGX box.

The following are some numbers:

|                      | Overall           | Data Read       | Data Preprocessing | Hashing Vectorizer | Tf-idf Transformer | Runs |
|----------------------|-------------------|-----------------|--------------------|--------------------|--------------------|------|
| **Scikit-Learn**         | 1401.311 &#177; 1.498 | 29.557 &#177; 0.991 | 915.171 &#177; 0.060   | 228.228 &#177; 1.535   | 228.35 &#177; 0.89     | 2    |
| **Apache Spark**         | 108.830 &#177; 0.677  | 0.07&#177;0.013     | 0.122 &#177; 0.009     | 0.024 &#177; 0.002     | 108.614 &#177; 0.671   | 5    |
| **cuML + Dask (4 GPUs)** | 9.948 &#177; 1.574    | 0.804 &#177; 0.437  | 4.743 &#177; 0.650     | 3.751 &#177; 0.509     | 0.519 &#177; 0.092     | 5    |

**NOTE:** The intermediate latency numbers in Apache Spark are generally garbage except the *Tf-idf Transformer* column, since that is where the computation across all partitions triggers in the lazy Spark pipeline. 

---

Comparing the end to end latency between Apache Spark and cuML + Dask pipeline, we see a **9.5x** speedup with the latter with just 4 GPUs !! 😎 😎
