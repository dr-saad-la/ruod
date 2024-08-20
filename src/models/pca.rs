#![allow(unused_imports)]
#![allow(unused)]

use nalgebra::{DMatrix, SVD};
use ndarray::{s, Array2, Axis, Zip};
use ndarray_linalg::MatrixLayout;
use std::error::Error;

/// A struct for performing PCA-based outlier detection.
///
/// This struct implements a PCA-based method to detect outliers in a dataset.
/// It reduces the dimensionality of the data and identifies outliers based on
/// their distance from the principal components.
pub struct PCAOutlierDetector {
    n_components: Option<usize>,
    n_selected_components: Option<usize>,
    contamination: f64,
    copy: bool,
    whiten: bool,
    standardize: bool,
    components: Option<Array2<f64>>,
    explained_variance: Option<Array2<f64>>,
    decision_scores: Option<Array2<f64>>,
}

impl PCAOutlierDetector {
    /// Constructs a new `PCAOutlierDetector` with the specified parameters.
    ///
    /// # Parameters
    /// - `n_components`: The number of components to keep after dimensionality reduction.
    /// - `n_selected_components`: The number of components to use for outlier detection.
    /// - `contamination`: The proportion of the dataset to be considered as outliers.
    /// - `copy`: Whether to copy the data before fitting.
    /// - `whiten`: Whether to whiten the components after fitting.
    /// - `standardize`: Whether to standardize the data before fitting.
    ///
    /// # Returns
    /// A new `PCAOutlierDetector` instance.
    pub fn new(
        n_components: Option<usize>,
        n_selected_components: Option<usize>,
        contamination: f64,
        copy: bool,
        whiten: bool,
        standardize: bool,
    ) -> Self {
        PCAOutlierDetector {
            n_components,
            n_selected_components,
            contamination,
            copy,
            whiten,
            standardize,
            components: None,
            explained_variance: None,
            decision_scores: None,
        }
    }

    /// Fits the PCA model to the data and calculates decision scores.
    ///
    /// This method standardizes the data if the `standardize` flag is set, centers the data,
    /// performs Singular Value Decomposition (SVD) to compute the principal components,
    /// applies whitening if the `whiten` flag is set, and calculates the decision scores
    /// for each data point.
    ///
    /// # Parameters
    /// - `x`: A 2D array containing the data to be analyzed.
    ///
    /// # Returns
    /// A `Result` indicating success or failure.
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), Box<dyn Error>> {
        // Validate the input data
        self.validate_input(x)?;

        let (n_samples, n_features) = x.dim();
        let mut x_centered = x.clone();

        // Standardize the data if the flag is set
        if self.standardize {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let std_dev = x.std_axis(Axis(0), 0.0);
            x_centered = (*&x - &mean) / &std_dev;
        } else {
            // Compute mean and center the data
            let mean = x.mean_axis(Axis(0)).unwrap();
            x_centered = x - &mean;
        }

        // Convert ndarray to nalgebra's DMatrix
        let x_centered_mat =
            DMatrix::from_iterator(n_samples, n_features, x_centered.iter().cloned());

        // Perform Singular Value Decomposition (SVD) using nalgebra
        let svd = SVD::new(x_centered_mat, true, true);

        let mut vt = svd.v_t.unwrap(); // V^T matrix (right singular vectors)
        let singular_values = svd.singular_values;

        // Apply whitening if the flag is set
        if self.whiten {
            for (i, singular_value) in singular_values.iter().enumerate() {
                vt.row_mut(i).scale_mut(1.0 / singular_value);
            }
        }

        // Convert nalgebra matrices back to ndarray arrays
        let eigenvectors =
            Array2::from_shape_vec((n_features, n_features), vt.as_slice().to_vec())?;

        self.components = Some(eigenvectors);

        // Compute explained variance (singular values squared and normalized)
        let explained_variance_vec: Vec<f64> = singular_values
            .iter()
            .map(|&val| val * val / (n_samples as f64 - 1.0))
            .collect();

        // Convert explained variance to ndarray
        let explained_variance =
            Array2::from_shape_vec((explained_variance_vec.len(), 1), explained_variance_vec)?;

        self.explained_variance = Some(explained_variance);

        // Project the data to the selected principal components
        let n_components = self.n_selected_components.unwrap_or(n_features);
        let start_idx = n_features - n_components;
        let selected_components = self
            .components
            .as_ref()
            .unwrap()
            .slice(s![start_idx.., ..])
            .to_owned();

        let scores = x_centered.dot(&selected_components.t());

        // Compute the L2 norm for each row in `scores`
        let decision_scores = scores.mapv(|v| v.powi(2)).sum_axis(Axis(1)).mapv(f64::sqrt);

        // Reshape the 1D decision_scores to a 2D array with a single column
        let decision_scores_2d = decision_scores.insert_axis(Axis(1));

        // Assign the reshaped array to decision_scores
        self.decision_scores = Some(decision_scores_2d);
        Ok(())
    }

    /// Validates the input data before processing.
    fn validate_input(&self, x: &Array2<f64>) -> Result<(), Box<dyn Error>> {
        let (n_samples, n_features) = x.dim();

        // Check that there are more than one sample and feature
        if n_samples < 2 || n_features < 2 {
            return Err(format!(
                "Input data must have at least 2 samples and 2 features. Got {} samples and {} features.",
                n_samples, n_features
            ).into());
        }

        // Check for NaN values
        if x.iter().any(|&val| val.is_nan()) {
            return Err("Input data contains NaN values, which are not allowed.".into());
        }

        Ok(())
    }

    /// Computes outlier scores for the given data based on the fitted PCA model.
    ///
    /// This method projects the input data onto the selected principal components and calculates
    /// the anomaly scores based on the distance of the data points from the principal components.
    ///
    /// # Parameters
    /// - `x`: A 2D array containing the data to be analyzed.
    ///
    /// # Returns
    /// A `Result` containing the outlier scores as a 2D array, or an error if the model has not been fitted.
    pub fn make_outlier_scores(&self, x: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        // Validate the input data
        self.validate_input(x)?;

        // Ensure the model is fitted
        if self.components.is_none() || self.explained_variance.is_none() {
            return Err(
                "Model is not fitted. Ensure `fit` is called before `make_outlier_scores`.".into(),
            );
        }

        // Transform the input data if needed (e.g., standardization)
        let x_transformed = x.clone(); // Assuming no additional transformation is required

        // Project the data to the selected principal components
        let n_features = x.dim().1;
        let n_components = self.n_selected_components.unwrap_or(n_features);
        let start_idx = n_features - n_components;
        let selected_components = self
            .components
            .as_ref()
            .unwrap()
            .slice(s![start_idx.., ..])
            .to_owned();

        // Compute the projection
        let projections = x_transformed.dot(&selected_components.t());

        // Calculate the anomaly scores
        let mut anomaly_scores = Array2::zeros((x.dim().0, 1));

        Zip::from(anomaly_scores.rows_mut())
            .and(projections.rows())
            .for_each(|mut anomaly_score_row, projection_row| {
                let score = projection_row
                    .iter()
                    .zip(self.explained_variance.as_ref().unwrap().iter())
                    .map(|(x, w)| (x / w).powi(2))
                    .sum::<f64>()
                    .sqrt();
                anomaly_score_row[0] = score;
            });

        Ok(anomaly_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests the PCAOutlierDetector with a small dataset.
    ///
    /// This test fits the PCA model to a small dataset and computes outlier scores,
    /// verifying that the model behaves as expected.
    #[test]
    fn test_pca_outlier_detection() {
        // Create a small dataset (5 samples, 3 features)
        let data = arr2(&[
            [2.5, 2.4, 1.0],
            [0.5, 0.7, 0.1],
            [2.2, 2.9, 1.1],
            [1.9, 2.2, 1.5],
            [3.1, 3.0, 0.9],
        ]);

        // Initialize PCAOutlierDetector with some parameters
        let mut pca_detector = PCAOutlierDetector::new(Some(2), Some(1), 0.1, true, true, false);

        // Fit the model
        let fit_result = pca_detector.fit(&data);
        assert!(fit_result.is_ok());

        // Check if the components were set
        assert!(pca_detector.components.is_some());

        // Check if decision scores are calculated
        assert!(pca_detector.decision_scores.is_some());

        // Make outlier scores
        let outlier_scores_result = pca_detector.make_outlier_scores(&data);

        // Borrow the error if the result is not OK
        if let Err(ref err) = outlier_scores_result {
            println!("Error in make_outlier_scores: {:?}", err);
        }

        assert!(outlier_scores_result.is_ok());

        let outlier_scores = outlier_scores_result.unwrap();

        // Check the shape of the outlier scores (should be [5, 1])
        assert_eq!(outlier_scores.dim(), (5, 1));

        // Print outlier scores for manual inspection
        println!("Outlier scores:\n{:?}", outlier_scores);

        // Optional: Check specific values for correctness (if expected values are known)
        // Example:
        // let expected_scores = arr2(&[[score1], [score2], [score3], [score4], [score5]]);
        // assert!(outlier_scores.abs_diff_eq(&expected_scores, 1e-8));
    }
}
