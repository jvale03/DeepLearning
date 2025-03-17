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
│   ├── scraper/
│   │   ├── main.py
│   │   ├── category.py
│   │   ├── papers.py
│   │   ├── test.py
│   │   └── info.md
│   └── lexicalParser/
│       ├── fetchScienceData.py
│       └── convertCsv.py
│
├── datasets/
|
├── models/
|
├── src/
│   ├── data_processor/
│   │   ├── data_processor.py
│   │   └── data_processor_v2.py
│   ├── model_RNN/
│   │   ├── recurrent_neural_net.py
│   │   ├── test_model.py
│   │   ├── test_model.script.py
│   │   ├── visualization.py
│   │   ├── optimizer.py
│   │   ├── metrics.py
│   │   ├── losses.py
│   │   ├── layers.py
│   │   └── data.py
│   ├── model_DNN/
│   │   ├── deep_neural_net.py
│   │   ├── test_model.py
│   │   ├── visualization.py
│   │   ├── optimizer.py
│   │   ├── metrics.py
│   │   ├── losses.py
│   │   ├── layers.py
│   │   └── data.py
│   └── losses.py
│
├── submissions/
│   ├── submission_1/
│
├── requirements.txt
├── .gitignore
└── .gitattributes
```

## Files and Their Purpose

### `dataset-generator/scraper/`
This module is responsible for collecting research papers from arXiv. It starts by scraping category pages, extracting links to individual papers, and downloading them for further processing.

### `dataset-generator/lexicalParser/`
This module extracts relevant scientific information based on predefined keywords and converts the extracted data into a structured format, such as CSV, for further analysis.

### `datasets/`
Stores the datasets used by the neural networks for both training and testing.

### `models/`
Stores the trained models generated after running the neural networks.

### `src/data_processor/`
Handles text preprocessing, including tokenization and structuring of data, to prepare it for model training. It ensures the input data is clean and usable for deep learning models.

### `src/model_RNN/`
Implements an RNN-based deep learning model for processing scientific text. It includes components for training, evaluation, optimization, and visualization of results.

### `src/model_DNN/`
Provides an alternative deep learning approach using a DNN model. Like the RNN module, it includes functionalities for training, testing, and analyzing performance.

### `submissions/`
All notebooks subsmissions and the .csv generated files.
