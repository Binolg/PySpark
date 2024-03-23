# Big Data Project Showcase

## Description

This repository serves as a showcase for a school project developed using PySpark for Big Data processing. It aims to demonstrate the practical application of PySpark and related technologies in handling large datasets.

### Structure
The contents are divided into 6 stages:
- Performance Analysis: Comparing dataset formats on different parameters
- Pre-Process Data: Preliminary data cleaning and structuring
- Data Exploration: Generic exploration to understand the data and its variables
- Data Processing: Construction of the pipeline and its various operations
- Models Performance: Testing different models for various dataset iterations
- Deployment: Local deployment with Streamlit to showcase the best model in action

### Dataset
The dataset used is artificialy generated, to simulate ban data in regards to what might be useful to determinate a customer's credit score.

#### Source
The original dataset can be found in [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).

## Simplified PySpark Usage with Docker on WSL

To streamline the PySpark setup process without the need for Java installation or virtual machine usage, Docker was employed within the Windows Subsystem for Linux (WSL) environment.

Using the official [pyspark-notebook image](https://quay.io/repository/jupyter/pyspark-notebook), a container was created. Ports 8888 (for the Jupyter notebook) and 8502 (for Streamlit) were exposed for easier access. Additionally, a host directory was mounted into the container to facilitate file access.

Although Nvidia GPU usage was configured for potential utilization, it was ultimately not required for the project's purposes.

This setup provides a straightforward and efficient environment for PySpark development and experimentation, enhancing productivity and ease of use.