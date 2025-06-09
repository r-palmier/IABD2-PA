// Trait Model local, défini dans ce fichier pour autonomie complète
pub trait Model {
    type Input;
    type Output;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>);
    fn predict(&self, x: &Self::Input) -> Self::Output;
}

// Struct LinearRegression avec poids (biais en position 0)
pub struct LinearRegression {
    pub weights: Vec<f64>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression { weights: Vec::new() }
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }
}

impl Model for LinearRegression {
    type Input = Vec<f64>;
    type Output = f64;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        let n_samples = x.len();
        let n_features = x[0].len();

        let mut mat_x = vec![vec![1.0; n_features + 1]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_features {
                mat_x[i][j + 1] = x[i][j];
            }
        }

        let xt = transpose(&mat_x);
        let xtx = matmul(&xt, &mat_x);
        let xtx_inv = inverse(&xtx).expect("Matrice X^T X non inversible.");
        let xty = matvecmul(&xt, y);

        self.weights = matvecmul(&xtx_inv, &xty);
    }

    fn predict(&self, x: &Self::Input) -> Self::Output {
        if self.weights.is_empty() {
            panic!("Modèle non entraîné");
        }
        let mut sum = self.weights[0];
        for i in 0..x.len() {
            sum += self.weights[i + 1] * x[i];
        }
        sum
    }
}

// Fonctions utilitaires locales

fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let rows_b = b.len();
    let cols_b = b[0].len();

    assert_eq!(cols_a, rows_b, "Dimensions incompatibles pour multiplication");

    let mut result = vec![vec![0.0; cols_b]; rows_a];
    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

fn matvecmul(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    assert_eq!(cols, vector.len(), "Dimensions incompatibles pour multiplication matrice-vecteur");

    let mut result = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += matrix[i][j] * vector[j];
        }
        result[i] = sum;
    }
    result
}

fn inverse(matrix: &Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    assert_eq!(n, matrix[0].len(), "La matrice doit être carrée");

    let mut a = matrix.clone();
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }

    for i in 0..n {
        let mut pivot = a[i][i];
        if pivot.abs() < 1e-12 {
            let mut found = false;
            for r in (i + 1)..n {
                if a[r][i].abs() > 1e-12 {
                    a.swap(i, r);
                    inv.swap(i, r);
                    pivot = a[i][i];
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
        for j in 0..n {
            a[i][j] /= pivot;
            inv[i][j] /= pivot;
        }
        for r in 0..n {
            if r != i {
                let factor = a[r][i];
                for c in 0..n {
                    a[r][c] -= factor * a[i][c];
                    inv[r][c] -= factor * inv[i][c];
                }
            }
        }
    }
    Some(inv)
}
