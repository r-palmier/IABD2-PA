use pyo3::prelude::*;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Simple binary perceptron using Rosenblatt's rule.
#[pyclass]
pub struct Perceptron {
    /// Weight vector including bias as the last element.
    weights: Vec<f32>,
}

#[pymethods]
impl Perceptron {
    #[new]
    fn new() -> Self {
        Self { weights: Vec::new() }
    }

    fn fit(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>, iterations: usize) -> PyResult<()> {
        if x.is_empty() {
            return Ok(());
        }
        let n_features = x[0].len();
        if self.weights.is_empty() {
            self.weights = vec![0.0; n_features + 1];
        }
        let lr = 1.0_f32;
        for _ in 0..iterations {
            for (xi, &target) in x.iter().zip(y.iter()) {
                let mut activation = self.weights[n_features];
                for j in 0..n_features {
                    activation += self.weights[j] * xi[j];
                }
                let predicted = if activation >= 0.0 { 1.0 } else { -1.0 };
                if (predicted - target).abs() > f32::EPSILON {
                    for j in 0..n_features {
                        self.weights[j] += lr * target * xi[j];
                    }
                    self.weights[n_features] += lr * target;
                }
            }
        }
        Ok(())
    }

    fn predict(&self, x: Vec<f32>) -> PyResult<f32> {
        if self.weights.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Model has not been trained"));
        }
        let n_features = self.weights.len() - 1;
        if x.len() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Input dimension does not match model"));
        }
        let mut activation = self.weights[n_features];
        for j in 0..n_features {
            activation += self.weights[j] * x[j];
        }
        Ok(if activation >= 0.0 { 1.0 } else { -1.0 })
    }
}

#[pyclass]
#[derive(Default)]
pub struct RBFN {
    centers: Option<Array2<f64>>,
    weights: Option<Array1<f64>>,
    gamma: f64,
}

#[pymethods]
impl RBFN {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>, n_centers: usize) -> PyResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty training data"));
        }
        let n_samples = x.len();
        let n_features = x[0].len();
        let x_flat: Vec<f64> = x.into_iter().flatten().collect();
        let x_arr = Array2::from_shape_vec((n_samples, n_features), x_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let y_arr = Array1::from_vec(y);

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());
        let indices = indices.into_iter().take(n_centers).collect::<Vec<_>>();
        let centers = x_arr.select(Axis(0), &indices);

        let k = centers.nrows();
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                let diff = &centers.row(i) - &centers.row(j);
                let dist = diff.dot(&diff).sqrt();
                sum += dist;
                count += 1;
            }
        }
        let sigma = if count > 0 { sum / count as f64 } else { 1.0 };
        self.gamma = 1.0 / (2.0 * sigma * sigma);

        let mut phi = Array2::<f64>::zeros((n_samples, k));
        for i in 0..n_samples {
            for j in 0..k {
                let diff = &x_arr.row(i) - &centers.row(j);
                let dist_sq = diff.dot(&diff);
                phi[[i, j]] = (-self.gamma * dist_sq).exp();
            }
        }

        let phi_t = phi.t();
        let a = phi_t.dot(&phi);
        let b = phi_t.dot(&y_arr);
        let w = ndarray_lstsq(&a, &b)?;

        self.centers = Some(centers);
        self.weights = Some(w);
        Ok(())
    }

    fn predict(&self, x: Vec<f64>) -> PyResult<f64> {
        let centers = self.centers.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not trained"))?;
        let weights = self.weights.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not trained"))?;
        if x.len() != centers.ncols() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Input dimension mismatch"));
        }
        let x_arr = Array1::from_vec(x);
        let k = centers.nrows();
        let mut phi = Array1::<f64>::zeros(k);
        for j in 0..k {
            let diff = &x_arr - &centers.row(j);
            let dist_sq = diff.dot(&diff);
            phi[j] = (-self.gamma * dist_sq).exp();
        }
        Ok(phi.dot(weights))
    }
}

fn ndarray_lstsq(a: &Array2<f64>, b: &Array1<f64>) -> PyResult<Array1<f64>> {
    let (n, m) = a.dim();
    let mut w = Array1::<f64>::zeros(m);
    for i in 0..m {
        if a[[i, i]].abs() < 1e-8 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Singular matrix or zero diagonal"));
        }
        w[i] = b[i] / a[[i, i]];
    }
    Ok(w)
}

#[pyfunction]
fn sum_vector(v: Vec<f32>) -> f32 {
    v.iter().sum()
}

#[pyfunction]
fn rbf_distance(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Vectors must have same length"));
    }
    let dist_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Ok(dist_sq.sqrt())
}

#[pymodule]
fn mylib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_vector, m)?)?;
    m.add_function(wrap_pyfunction!(rbf_distance, m)?)?;
    m.add_class::<Perceptron>()?;
    m.add_class::<RBFN>()?;
    Ok(())
}
