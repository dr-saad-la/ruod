use pyo3::prelude::*;
use ruod::models::pca::PCAOutlierDetector;

#[pymodule]
fn ruod(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PCAOutlierDetector>()?;
    Ok(())
}
