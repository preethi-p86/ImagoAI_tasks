import torch
import visualization
import model
import data_loader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import logger

def evaluate():
    """Evaluates the model and logs results."""
    train_loader, test_loader = data_loader.load_data()
    model, _, _ = model.get_model()
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Flatten lists for scoring functions
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # visualize actual and predicted values
    visualization.plot_predictions()

    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_true, y_pred)  # R^2 Score

    # Log results
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test R^2: {r2:.4f}")
