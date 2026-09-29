# Processed data folder

The modeling code intentionally performs imputation, scaling and encoding **inside scikit-learn Pipelines after the train/test split**. This avoids data leakage. Therefore a globally preprocessed modeling CSV is not committed here by default.

If you create exported transformed data for learning or reporting, place it in this folder, but do not use statistics computed from the full dataset to preprocess the training/test split.
