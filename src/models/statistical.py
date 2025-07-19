import numpy as np
import pandas as pd
import warnings

from sklearn.covariance import MinCovDet
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors # Added NearestNeighbors

warnings.filterwarnings("ignore")

def z_score_method(dataframe, anomaly_column, description_column, threshold, dataframe_eval):
    dataframe_no_anomaly = dataframe[dataframe[anomaly_column] == 0]
    dataframe_no_anomaly.drop(columns=[anomaly_column, description_column], inplace=True)

    dataframe_test = dataframe_eval.drop(columns=[anomaly_column, description_column])

    means = dataframe_no_anomaly.mean(axis=0)
    stds = dataframe_no_anomaly.std(axis=0)
    stds[stds == 0] = 1e-9 # Replace zero std with a small number

    def _calculate_zscores_and_labels(X_data: pd.DataFrame, means: pd.Series, stds: pd.Series):
        z_scores = np.abs((X_data - means) / stds)
        # Anomaly score: Max Z-score across features for each data point
        # Higher score = more anomalous
        scores = z_scores.max(axis=1).values
        # Binary labels based on threshold: 1 if any feature's Z-score > threshold
        labels = (z_scores.max(axis=1) > threshold).astype(int)
        return scores, labels

    _, y_pred = _calculate_zscores_and_labels(dataframe_test, means, stds)
    y_test = np.array([0 if idx==0 else 1 for idx in dataframe_eval[anomaly_column].values.ravel()])
    return y_pred, y_test

def multivariate_gaussian_model(dataframe, anomaly_column, description_column, contamination, robust_covariance, dataframe_eval):
    dataframe_no_anomaly = dataframe[dataframe[anomaly_column] == 0]
    dataframe_no_anomaly.drop(columns=[anomaly_column, description_column], inplace=True)

    dataframe_test = dataframe_eval.drop(columns=[anomaly_column, description_column])

    mu = dataframe_no_anomaly.mean(axis=0).values
    sigma = dataframe_no_anomaly.cov().values
    if robust_covariance:
        # Robust Covariance Estimation (e.g., Minimum Covariance Determinant)
        # More robust to outliers in the training data itself.
        cov_estimator = MinCovDet(random_state=42).fit(dataframe_no_anomaly)
        sigma = cov_estimator.covariance_

    # Check for singularity of covariance matrix
    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular. Adding small regularization.")
        sigma += np.eye(sigma.shape[0]) * 1e-6 # Add a small value to diagonal for regularization
        sigma_inv = np.linalg.inv(sigma)
    
    def _convert_scores_to_labels(scores: np.ndarray, contamination: float) -> np.ndarray:
        """
        Converts anomaly scores to binary labels (1 for anomaly, 0 for normal).
        Higher scores typically indicate higher anomaly likelihood.
        """
        # Determine threshold based on the desired contamination (proportion of anomalies)
        # Smaller scores are more normal, larger scores are more anomalous.
        # So, we want to find the (1 - contamination) percentile score.
        threshold = np.percentile(scores, 100 * (1 - contamination))
        labels = (scores > threshold).astype(int)
        return labels, threshold

    def _calculate_mvg_scores_and_labels(X_data: pd.DataFrame, mu: np.ndarray, sigma_inv: np.ndarray, contamination: float):
        X_data_np = X_data.values
        num_features = X_data_np.shape[1]

        # Mahalanobis distance squared (higher means more anomalous)
        # D^2 = (x - mu)^T * Sigma_inv * (x - mu)
        diff = X_data_np - mu
        mahalanobis_dist_sq = np.diag(diff @ sigma_inv @ diff.T)

        # Anomaly score: Negative Log Likelihood (higher score = more anomalous)
        # This is proportional to Mahalanobis distance squared for Gaussian
        scores = mahalanobis_dist_sq # Use Mahalanobis distance squared directly as score

        # Convert scores to labels based on contamination
        labels, threshold = _convert_scores_to_labels(scores, contamination)

        # Alternative thresholding: using Chi-squared distribution (p-value)
        # Assuming the scores are Mahalanobis distance squared and approximately chi-squared distributed
        # You could also use a p-value threshold (e.g., 0.01 for 1% anomalies)
        # threshold_chi2 = chi2.ppf(1 - contamination, df=num_features)
        # labels_chi2 = (scores > threshold_chi2).astype(int)
        # print(f"    MVG Chi-squared threshold: {threshold_chi2:.2f}, count: {np.sum(labels_chi2)}")

        return scores, labels

    _, y_pred = _calculate_mvg_scores_and_labels(dataframe_test, mu, sigma_inv, contamination)
    y_test = np.array([0 if idx==0 else 1 for idx in dataframe_eval[anomaly_column].values.ravel()])
    return y_pred, y_test

def interquartile_range_method(dataframe, anomaly_column, description_column, iqr_multiplier, dataframe_eval):
    dataframe_no_anomaly = dataframe[dataframe[anomaly_column] == 0]
    dataframe_no_anomaly.drop(columns=[anomaly_column, description_column], inplace=True)

    dataframe_test = dataframe_eval.drop(columns=[anomaly_column, description_column])

    # Fit: Calculate Q1, Q3, and IQR for each feature on X_train
    q1 = dataframe_no_anomaly.quantile(0.25, axis=0)
    q3 = dataframe_no_anomaly.quantile(0.75, axis=0)
    iqr = q3 - q1

    # Define lower and upper bounds for each feature
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    def _calculate_iqr_scores_and_labels(X_data: pd.DataFrame, lower_bound: pd.Series, upper_bound: pd.Series):
        # Calculate distance from bounds. Positive means outside, negative means inside.
        # Max distance across features will be the score.
        scores = np.zeros(X_data.shape[0])
        labels = np.zeros(X_data.shape[0], dtype=int)

        # For each feature, find points outside bounds
        # And calculate max "outlierness" score
        for i, col in enumerate(X_data.columns):
            outlier_low = X_data[col] < lower_bound[col]
            outlier_high = X_data[col] > upper_bound[col]

            # Assign labels (1 if outlier for this feature)
            labels = np.maximum(labels, (outlier_low | outlier_high).astype(int))

            # Calculate "outlierness" score: distance from boundary
            # If value is below lower_bound, score is (lower_bound - value)
            # If value is above upper_bound, score is (value - upper_bound)
            # If inside, score is 0
            feature_scores = np.zeros(X_data.shape[0])
            feature_scores[outlier_low] = lower_bound[col] - X_data[col][outlier_low]
            feature_scores[outlier_high] = X_data[col][outlier_high] - upper_bound[col]

            scores = np.maximum(scores, feature_scores) # Take max score across all features

        # We prioritize the labels derived from the direct IQR multiplier
        # But we return the contamination-based labels for consistency in overall evaluation comparison
        # Let's return the direct labels derived from the multiplier, as that's the primary
        # way IQR is often used. The score is provided for plots.
        return scores, labels

    _, y_pred = _calculate_iqr_scores_and_labels(dataframe_test, lower_bound, upper_bound)
    y_test = np.array([0 if idx==0 else 1 for idx in dataframe_eval[anomaly_column].values.ravel()])
    return y_pred, y_test

def percentile_method(dataframe, anomaly_column, description_column, lower_percentile, upper_percentile, dataframe_eval):
    dataframe_no_anomaly = dataframe[dataframe[anomaly_column] == 0]
    dataframe_no_anomaly.drop(columns=[anomaly_column, description_column], inplace=True)

    dataframe_test = dataframe_eval.drop(columns=[anomaly_column, description_column])

    # Fit: Calculate lower and upper percentile bounds for each feature on X_train
    lower_bounds = dataframe_no_anomaly.quantile(lower_percentile / 100.0, axis=0)
    upper_bounds = dataframe_no_anomaly.quantile(upper_percentile / 100.0, axis=0)

    def _calculate_percentile_scores_and_labels(X_data: pd.DataFrame, lower_bounds: pd.Series, upper_bounds: pd.Series):
        scores = np.zeros(X_data.shape[0])
        labels = np.zeros(X_data.shape[0], dtype=int)

        for i, col in enumerate(X_data.columns):
            outlier_low = X_data[col] < lower_bounds[col]
            outlier_high = X_data[col] > upper_bounds[col]

            labels = np.maximum(labels, (outlier_low | outlier_high).astype(int))

            # Calculate "outlierness" score: distance from boundary
            feature_scores = np.zeros(X_data.shape[0])
            feature_scores[outlier_low] = lower_bounds[col] - X_data[col][outlier_low]
            feature_scores[outlier_high] = X_data[col][outlier_high] - upper_bounds[col]
            scores = np.maximum(scores, feature_scores)

        return scores, labels # Return direct labels based on percentile bounds

    _, y_pred = _calculate_percentile_scores_and_labels(dataframe_test, lower_bounds, upper_bounds)
    y_test = np.array([0 if idx==0 else 1 for idx in dataframe_eval[anomaly_column].values.ravel()])
    return y_pred, y_test

def k_nearest_neighbors(dataframe, anomaly_column, description_column, n_neighbors, contamination, metric, dataframe_eval):
    dataframe_no_anomaly = dataframe[dataframe[anomaly_column] == 0]
    dataframe_no_anomaly.drop(columns=[anomaly_column, description_column], inplace=True)

    dataframe_test = dataframe_eval.drop(columns=[anomaly_column, description_column])

    # Fit NearestNeighbors model on X_train. This builds the KD-tree or Ball-tree.
    # The 'n_neighbors' parameter here is for the query, not for the fit.
    # We are interested in distances to n_neighbors + 1 because the first neighbor is always itself (distance 0).
    # So we need the (n_neighbors + 1)th distance to be the k-th nearest distinct neighbor.
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1)
    nn_model.fit(dataframe_no_anomaly)

    def _convert_scores_to_labels(scores: np.ndarray, contamination: float, score_lower_is_anomaly: bool = False) -> tuple[np.ndarray, float]:
        """
        Converts anomaly scores to binary labels (1 for anomaly, 0 for normal).
        Args:
            scores (np.ndarray): Array of anomaly scores.
            contamination (float): Estimated proportion of anomalies.
            score_lower_is_anomaly (bool): If True, lower scores mean more anomalous.
                                        If False (default), higher scores mean more anomalous.
        Returns:
            tuple: (labels, threshold)
        """
        if score_lower_is_anomaly:
            # For methods where lower score is more anomalous (e.g., KDE log-likelihood)
            # We want to find the contamination-percentile score as the threshold.
            threshold = np.percentile(scores, 100 * contamination)
            labels = (scores < threshold).astype(int)
        else:
            # For methods where higher score is more anomalous
            threshold = np.percentile(scores, 100 * (1 - contamination))
            labels = (scores > threshold).astype(int)
        return labels, threshold

    def _calculate_knn_scores_and_labels(X_data: pd.DataFrame, nn_model: NearestNeighbors, contamination: float):
        # distances.shape = (n_samples, n_neighbors + 1)
        # indices.shape = (n_samples, n_neighbors + 1)
        distances, _ = nn_model.kneighbors(X_data)

        # Anomaly score: distance to the k-th nearest neighbor (index n_neighbors after removing self-distance)
        # Note: distances are already sorted. So the (n_neighbors)th column (0-indexed) is the distance
        # to the k-th neighbor (excluding self, as the first column is distance to self).
        scores = distances[:, n_neighbors] # Distance to the (k+1)th neighbor if including self, effectively k-th neighbor

        # Convert scores to labels based on contamination
        # Higher score means further from neighbors = more anomalous
        labels, _ = _convert_scores_to_labels(scores, contamination, score_lower_is_anomaly=False)

        return scores, labels
    
    _, y_pred = _calculate_knn_scores_and_labels(dataframe_test, nn_model, contamination)
    y_test = np.array([0 if idx==0 else 1 for idx in dataframe_eval[anomaly_column].values.ravel()])
    return y_pred, y_test

def local_outlier_factor(dataframe, anomaly_column, description_column, n_neighbors, contamination, dataframe_eval):
    dataframe_no_anomaly = dataframe[dataframe[anomaly_column] == 0]
    dataframe_no_anomaly.drop(columns=[anomaly_column, description_column], inplace=True)

    dataframe_test = dataframe_eval.drop(columns=[anomaly_column, description_column])

    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    model.fit(dataframe_no_anomaly) # Fit on the training data

    y_pred = (model.predict(dataframe_test) == -1).astype(int)
    y_test = np.array([0 if idx==0 else 1 for idx in dataframe_eval[anomaly_column].values.ravel()])
    return y_pred, y_test
