#!/usr/bin/env bash
set -e
python src/linear_experiments.py
python src/train_models.py
python src/explain_model.py
python src/predict.py
