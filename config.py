# config.py

DATA_PATH = "data/hyperspectral_data.csv"
MODEL_PATH = "models/mycotoxin_regressor.pth"

TEST_SIZE = 0.2
RANDOM_SEED = 42

NORMALIZATION_METHOD = "standard"  # Options: "minmax", "standard"
IMPUTATION_METHOD = "mean"  # Options: "mean", "median"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001