#![allow(unused_imports)]
#![allow(unused)]

use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::{s, Array2, Axis};
use ndarray_linalg::{Lapack, Norm, SVD};
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
    components: Option<Array2<f64>>,
    explained_variance: Option<Array2<f64>>,
    decision_scores: Option<Array2<f64>>,
}

impl PCAOutlierDetector {
    /// Constructs a new `PCAOutlierDetector` with the specified parameters.
    pub fn new(
        n_components: Option<usize>,
        n_selected_components: Option<usize>,
        contamination: f64,
        copy: bool,
        whiten: bool,
    ) -> Self {
        PCAOutlierDetector {
            n_components,
            n_selected_components,
            contamination,
            copy,
            whiten,
            components: None,
            explained_variance: None,
            decision_scores: None,
        }
    }

    /// fit method
    pub fn fit() {
        // implement the fit method here
    }

    /// make outlier scores

    pub fn make_outlier_scores() {
        // code here
    }
}
