### Benchmarking latency of a NLP tf-idf pipeline across Sklearn, cuML and Spark using Hashing Vectorizer. 

**Note:** All tests are done from within a `rapidsai/rapidsai-nightly:21.08-cuda11.2-runtime-ubuntu18.04-py3.8` container. PySpark version is 3.1.2.

For the study, we have used a subset of the [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), which is a dataset of customer reviews of products available on the Amazon platform. We particularly use only the reviews of PC products in this study. The data is available in both tab separated (`tsv`) and `parquet` file format. We will use the parquet file format. For the test however, downloading the parquet files in the notebook is extremely slow. Hence we have downloaded the parquet files in the DGX box before running the tests and then read from disk in the notebooks.

The dataset can be downloaded locally by the following (assuming you have `aws-cli` set up):
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
| Scikit-Learn         | 1180.900 &#177; 3.531 | 16.990 &#177; 0.677 | 910.171 &#177; 1.612   | 228.228 &#177; 1.535   | 13.10 &#177; 0.715     | 2    |
| Apache Spark         | 110.309 &#177; 1.043  | 0.062 &#177; 0.001  | 0.107 &#177; 0.004     | 0.0145 &#177; 0        | 110.126 &#177; 1.048   | 5    |
| cuML + Dask (4 GPUs) | 10.017 &#177; 1.654   | 0.742 &#177; 0.399  | 4.837 &#177; 0.476     | 3.866 &#177; 0.405     | 0.460 &#177; 0.161     | 5    |

**NOTE:** The intermediate latency numbers in Apache Spark are generally garbage except the *Tf-idf Transformer* column, since that is where the computation across all partitions triggers in the lazy Spark pipeline. 

---

Comparing the end to end latency between Apache Spark and cuML + Dask pipeline, we see a **9.5x** speedup with the latter with just 4 GPUs !! 😎 😎
