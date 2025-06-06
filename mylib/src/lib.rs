use pyo3::prelude::*;

/// Simple binary perceptron using Rosenblatt's rule.
#[pyclass]
pub struct Perceptron {
    /// Weight vector including bias as the last element.
    weights: Vec<f32>,
}

#[pymethods]
impl Perceptron {
    /// Create a new perceptron with no initialized weights.
    #[new]
    fn new() -> Self {
        Self { weights: Vec::new() }
    }

    /// Train the perceptron with the provided data for a number of iterations.
    ///
    /// `x` is a list of feature vectors and `y` the corresponding labels (-1.0 or 1.0).
    /// Weights are initialized to zero on the first call and include a bias term.
    fn fit(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>, iterations: usize) -> PyResult<()> {
        if x.is_empty() {
            return Ok(());
        }

        let n_features = x[0].len();
        if self.weights.is_empty() {
            self.weights = vec![0.0; n_features + 1];
        }

        let lr = 1.0_f32; // learning rate for Rosenblatt update

        for _ in 0..iterations {
            for (xi, &target) in x.iter().zip(y.iter()) {
                let mut activation = self.weights[n_features]; // bias weight
                for j in 0..n_features {
                    activation += self.weights[j] * xi[j];
                }

                let predicted = if activation >= 0.0 { 1.0 } else { -1.0 };

                if (predicted - target).abs() > f32::EPSILON {
                    // Update rule: w_i += lr * target * x_i, bias += lr * target
                    for j in 0..n_features {
                        self.weights[j] += lr * target * xi[j];
                    }
                    self.weights[n_features] += lr * target;
                }
            }
        }
        Ok(())
    }

    /// Predict the label for a single feature vector.
    fn predict(&self, x: Vec<f32>) -> PyResult<f32> {
        if self.weights.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model has not been trained",
            ));
        }

        let n_features = self.weights.len() - 1;
        if x.len() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input dimension does not match model",
            ));
        }

        let mut activation = self.weights[n_features];
        for j in 0..n_features {
            activation += self.weights[j] * x[j];
        }

        Ok(if activation >= 0.0 { 1.0 } else { -1.0 })
    }
}

#[pyfunction]
fn sum_vector(v: Vec<f32>) -> f32 {
    v.iter().sum()
}

#[pymodule]
fn mylib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_vector, m)?)?;
    m.add_class::<Perceptron>()?;
    Ok(())
}
