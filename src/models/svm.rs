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

pub struct SVM {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    epochs: usize,
    lambda: f64,
    gamma: f64,
    use_kernel: bool,
    support_vectors: Vec<Vec<f64>>,
    support_labels: Vec<i8>,
}

impl SVM {
    pub fn new(input_dim: usize, learning_rate: f64, epochs: usize, lambda: f64, gamma: f64) -> Self {
        SVM {
            weights: vec![0.0; input_dim],
            bias: 0.0,
            learning_rate,
            epochs,
            lambda,
            gamma,
            use_kernel: false,
            support_vectors: Vec::new(),
            support_labels: Vec::new(),
        }
    }

    pub fn enable_kernel(&mut self, use_kernel: bool) {
        self.use_kernel = use_kernel;
    }

    fn rbf_kernel(&self, x1: &Vec<f64>, x2: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x1.iter().zip(x2.iter()).map(|(a,b)| (a-b).powi(2)).sum();
        (-self.gamma * dist_sq).exp()
    }

    fn shuffle_data(&self, x: &mut Vec<Vec<f64>>, y: &mut Vec<i8>) {
        let mut rng = SimpleRng::new(123456);
        let n = x.len();
        for i in (1..n).rev() {
            let j = rng.gen_range_usize(0, i + 1);
            x.swap(i, j);
            y.swap(i, j);
        }
    }

pub fn train(&mut self, x: &Vec<Vec<f64>>, y: &Vec<i8>) {
    let mut x_data = x.clone();
    let mut y_data = y.clone();

    for epoch in 0..self.epochs {
        self.shuffle_data(&mut x_data, &mut y_data);

        for (xi, &yi) in x_data.iter().zip(y_data.iter()) {
            let condition = if self.use_kernel {
                self.support_vectors.iter().zip(self.support_labels.iter())
                    .map(|(sv, &label)| label as f64 * self.rbf_kernel(xi, sv))
                    .sum::<f64>() + self.bias
            } else {
                self.weights.iter().zip(xi.iter()).map(|(w, xi)| w * xi).sum::<f64>() + self.bias
            };

            let cond = yi as f64 * condition >= 1.0;

            if cond {
                if !self.use_kernel {
                    for i in 0..self.weights.len() {
                        self.weights[i] -= self.learning_rate * 2.0 * self.lambda * self.weights[i];
                    }
                }
            } else {
                if self.use_kernel {
                    self.support_vectors.push(xi.clone());
                    self.support_labels.push(yi);
                } else {
                    for i in 0..self.weights.len() {
                        self.weights[i] += self.learning_rate * (yi as f64 * xi[i] - 2.0 * self.lambda * self.weights[i]);
                    }
                    self.bias += self.learning_rate * yi as f64;
                }
            }
        }

        // Calcul de la loss hinge sur tout le dataset pour suivi
        let loss = x.iter().zip(y.iter()).map(|(xi, &yi)| {
            let margin = yi as f64 * (self.weights.iter().zip(xi.iter()).map(|(w, xi)| w * xi).sum::<f64>() + self.bias);
            (1.0 - margin).max(0.0)
        }).sum::<f64>() / x.len() as f64;

        if epoch % 100 == 0 {
            println!("Epoch {}: hinge loss = {:.6}", epoch, loss);
        }
    }
}


    pub fn predict_raw(&self, x: &Vec<f64>) -> f64 {
        if self.use_kernel {
            self.support_vectors.iter().zip(self.support_labels.iter())
                .map(|(sv, &label)| label as f64 * self.rbf_kernel(x, sv))
                .sum::<f64>() + self.bias
        } else {
            self.weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + self.bias
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> i8 {
        if self.predict_raw(x) >= 0.0 { 1 } else { -1 }
    }

    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }
}

impl Model for SVM {
    type Input = Vec<f64>;
    type Output = i8;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        self.train(x, y);
    }

    fn predict(&self, x: &Self::Input) -> Self::Output {
        self.predict(x)
    }
}
