# Deep Learning Project

- PG55951 - João Vale
- PG55977 - Luís Borges
- PG56015 - Tomás Oliveira
- PG55948 - Hugo Ramos
- PG55974 - Leonardo Barroso

This project consists of various scripts and modules for scraping research papers, processing data, and implementing deep learning models.

## Directory Structure

```
Project/
│
├── dataset-generator/
│   ├── dataset-cleaner/
│   ├── lexicalParser/
│   └── scraper/
│
├── datasets/
│
├── models/
│
├── src/
│   ├── CustomModels/
│   │   ├── data_processor/
│   │   ├── model_RNN/
│   │   ├── model_DNN/
│   ├── TensorflowModels/
│
├── submissions/
│   ├── submission_1/
│   ├── submission_2/
│   └── submission_3/
```

## Files and Their Purpose

### `dataset-generator/dataset-cleaner/`
Contains scripts for cleaning the training and validation datasets.

### `dataset-generator/lexicalParser/`
This module extracts relevant scientific information based on predefined keywords and converts the extracted data into a structured format, such as CSV, for further analysis.

### `dataset-generator/scraper/`
This module is responsible for collecting research papers from arXiv. It starts by scraping category pages, extracting links to individual papers, and downloading them for further processing and building a dataset.

### `datasets/`
Contains every dataset used in the whole repository.

### `models/`
Stores the trained models generated after running the neural networks.

### `src/CustomModels/`
Contains models created from scratch without using TensorFlow or other deep learning libraries. These models are implemented using fundamental machine learning techniques and custom neural network architectures.

### `src/CustomModels/data_processor/`
Handles text preprocessing, including tokenization and structuring of data, to prepare it for model training. It ensures the input data is clean and usable for deep learning models.

### `src/CustomModels/model_DNN/`
Provides an alternative deep learning approach using a DNN model. Like the RNN module, it includes functionalities for training, testing, and analyzing performance.

### `src/CustomModels/model_RNN/`
Implements an RNN-based deep learning model for processing scientific text. It includes components for training, evaluation, optimization, and visualization of results.

### `src/TensorflowModels/`
Contains various TensorFlow-based models including RNN, LSTM, and BERT implementations for text classification tasks.

### `submissions/`
Contains all notebook submissions and the generated .csv files, split by directories, each corresponding to a specific submission date.
