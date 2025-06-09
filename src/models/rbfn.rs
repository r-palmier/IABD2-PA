pub trait Model {
    type Input;
    type Output;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>);
    fn predict(&self, x: &Self::Input) -> Self::Output;
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        (self.state as f64) / (u64::MAX as f64)
    }

    fn gen_range_usize(&mut self, min: usize, max: usize) -> usize {
        min + (self.next_f64() * ((max - min) as f64)) as usize
    }
}

pub struct RBFN {
    centers: Vec<Vec<f64>>,
    weights: Vec<f64>,
    gamma: f64,
    k: usize,
    max_kmeans_iter: usize,
}

impl RBFN {
    pub fn new(k: usize, gamma: f64, max_kmeans_iter: usize) -> Self {
        RBFN {
            centers: Vec::new(),
            weights: Vec::new(),
            gamma,
            k,
            max_kmeans_iter,
        }
    }

    pub fn train(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        assert!(x.len() >= self.k, "Pas assez d’exemples pour k centres");

        let mut rng = SimpleRng::new(123456789);

        // Initialisation aléatoire des centres parmi x
        let mut chosen_indices = Vec::new();
        while chosen_indices.len() < self.k {
            let idx = rng.gen_range_usize(0, x.len());
            if !chosen_indices.contains(&idx) {
                chosen_indices.push(idx);
            }
        }
        self.centers = chosen_indices.iter().map(|&i| x[i].clone()).collect();

        // K-means simplifié
        for _ in 0..self.max_kmeans_iter {
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.k];

            for xi in x.iter() {
                let (closest_idx, _) = self.centers.iter().enumerate()
                    .map(|(i, c)| (i, squared_euclidean_distance(xi, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                clusters[closest_idx].push(xi.clone());
            }

            let mut new_centers = Vec::new();
            for cluster in clusters.iter() {
                if cluster.is_empty() {
                    new_centers.push(vec![0.0; x[0].len()]);
                } else {
                    new_centers.push(mean_vector(cluster));
                }
            }

            if new_centers == self.centers {
                break;
            }
            self.centers = new_centers;
        }

        // Calcul matrice Phi (n_samples x k)
        let n_samples = x.len();
        let mut phi = vec![vec![0.0; self.k]; n_samples];
        for i in 0..n_samples {
            for j in 0..self.k {
                phi[i][j] = self.gaussian_rbf(&x[i], &self.centers[j]);
            }
        }

        // Calcul poids via pseudo-inverse : w = (Phi^T Phi)^-1 Phi^T y
        let phi_t = transpose(&phi);
        let phi_t_phi = matmul(&phi_t, &phi);
        let phi_t_phi_inv = inverse(&phi_t_phi).expect("Matrice Phi^T Phi non inversible.");
        let phi_t_y = matvecmul(&phi_t, y);
        let w = matvecmul(&phi_t_phi_inv, &phi_t_y);

        self.weights = w;
    }

    fn gaussian_rbf(&self, x: &Vec<f64>, c: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| (xi - ci).powi(2)).sum();
        (-self.gamma * dist_sq).exp()
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        self.centers.iter().zip(self.weights.iter())
            .map(|(c, w)| w * self.gaussian_rbf(x, c))
            .sum()
    }
}

// Fonctions utilitaires

fn squared_euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn mean_vector(vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let len = vectors[0].len();
    let mut sum = vec![0.0; len];
    for v in vectors {
        for i in 0..len {
            sum[i] += v[i];
        }
    }
    let n = vectors.len() as f64;
    for i in 0..len {
        sum[i] /= n;
    }
    sum
}

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

    assert_eq!(cols_a, rows_b, "Dimensions incompatibles");

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

    assert_eq!(cols, vector.len(), "Dimensions incompatibles");

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
