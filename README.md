## Goal

Automate the lifecycle of a machine learning model ( data -> model training -> deployment ) using the provided Python scripts.

Focus on _demonstrating automation_ and continuous update capabilities within a realistic (but minimal) MLOps workflow.

## Resources

- `get_data.py`
- `train_model.py`
- `evaluate_model.py`

## Tasks

1. Automatically generate and save new data to disk at regular intervals.
2. Train the model on new data, and save new parameters automatically.
3. "Deploy" the best model at any time by saving model weights to a production directory with version tracking.
