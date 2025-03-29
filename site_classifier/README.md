# Site Classification Project

A project to clean website data and train a classifier model.

## Directory Structure

- `data/`: Raw and processed data files
- `src/`: Source code for data cleaning and model training
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `models/`: Saved model weights and configurations
- `config/`: Configuration files

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Clean the data:
```bash
python src/clean_data.py
```

2. Train the model:
```bash
python src/train_model.py
```
