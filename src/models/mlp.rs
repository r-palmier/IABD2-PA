// Trait Model local pour interface commune
pub trait Model {
    type Input;
    type Output;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>);
    fn predict(&self, x: &Self::Input) -> Self::Output;
}

// Générateur pseudo-aléatoire simple (LCG)
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        (self.state as f64) / (u64::MAX as f64)
    }

    pub fn gen_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

// Structure représentant une couche dense avec activation tanh
pub struct Layer {
    weights: Vec<Vec<f64>>,  // poids : input_size x output_size
    biases: Vec<f64>,        // biais : output_size
    input_size: usize,
    output_size: usize,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> Self {
        let mut weights = vec![vec![0.0; output_size]; input_size];
        let biases = vec![0.0; output_size];

        for i in 0..input_size {
            for j in 0..output_size {
                weights[i][j] = rng.gen_range(-1.0, 1.0);
            }
        }

        Layer {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        for j in 0..self.output_size {
            let mut sum = self.biases[j];
            for i in 0..self.input_size {
                sum += input[i] * self.weights[i][j];
            }
            output[j] = sum.tanh();
        }
        output
    }

    pub fn activation_derivative(output: &Vec<f64>) -> Vec<f64> {
        output.iter().map(|o| 1.0 - o.powi(2)).collect()
    }
}

// Perceptron multi-couches complet
pub struct MLP {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl MLP {
    pub fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        let mut rng = SimpleRng::new(123456789);
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1], &mut rng));
        }

        MLP { layers, learning_rate }
    }

    // Propagation avant sur toutes les couches, retourne toutes les activations
    pub fn forward(&self, input: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut activations = Vec::new();
        let mut current_input = input.clone();

        for layer in &self.layers {
            let output = layer.forward(&current_input);
            activations.push(output.clone());
            current_input = output;
        }

        activations
    }

    // Entraînement par rétropropagation
    pub fn train(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<Vec<f64>>, epochs: usize) {
        for _ in 0..epochs {
            for (x, y) in x_train.iter().zip(y_train.iter()) {
                let activations = self.forward(x);
                self.backward(x, y, &activations);
            }
        }
    }

    // Rétropropagation et mise à jour des poids
    fn backward(&mut self, input: &Vec<f64>, target: &Vec<f64>, activations: &Vec<Vec<f64>>) {
        let mut errors = Vec::new();
        let output_activations = activations.last().unwrap();

        // Calcul de l’erreur en sortie
        let output_error: Vec<f64> = target.iter()
            .zip(output_activations.iter())
            .map(|(t, o)| t - o)
            .collect();

        errors.push(output_error);

        // Propagation des erreurs couche par couche à rebours
        for l in (1..self.layers.len()).rev() {
            let layer = &self.layers[l];
            let prev_layer = &self.layers[l - 1];
            let layer_error = &errors[0];

            let mut prev_error = vec![0.0; prev_layer.output_size];

            for i in 0..prev_layer.output_size {
                let mut sum = 0.0;
                for j in 0..layer.output_size {
                    sum += layer.weights[i][j] * layer_error[j];
                }
                prev_error[i] = sum;
            }
            errors.insert(0, prev_error);
        }

        // Mise à jour des poids et biais
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let input_activations = if i == 0 { input } else { &activations[i - 1] };
            let deriv = Layer::activation_derivative(&activations[i]);

            for j in 0..layer.output_size {
                let delta = errors[i][j] * deriv[j];
                layer.biases[j] += self.learning_rate * delta;

                for k in 0..layer.input_size {
                    layer.weights[k][j] += self.learning_rate * delta * input_activations[k];
                }
            }
        }
    }
}

impl Model for MLP {
    type Input = Vec<f64>;
    type Output = Vec<f64>;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        self.train(x, y, 1000);
    }

    fn predict(&self, x: &Self::Input) -> Self::Output {
        let activations = self.forward(x);
        activations.last().unwrap().clone()
    }
}
