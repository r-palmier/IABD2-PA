// Trait Model défini localement
pub trait Model {
    type Input;
    type Output;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>);
    fn predict(&self, x: &Self::Input) -> Self::Output;
}

// Struct représentant un Perceptron simple (classification linéaire)
pub struct LinearClassification {
    pub weights: Vec<f64>, // poids avec biais en position 0
}

impl LinearClassification {
    // Constructeur avec poids initialement vides
    pub fn new() -> Self {
        LinearClassification { weights: Vec::new() }
    }

    // Permet de fixer manuellement les poids (utile pour tests)
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    // Permet d’accéder aux poids
    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }
}

// Implémentation du trait Model pour LinearClassification
impl Model for LinearClassification {
    type Input = Vec<f64>;
    type Output = i8;

    // Apprentissage avec la règle de mise à jour de Rosenblatt
    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        let n_samples = x.len();
        let n_features = x[0].len();

        // Initialisation des poids à zéro, poids[0] = biais
        self.weights = vec![0.0; n_features + 1];

        let learning_rate = 0.1;
        let max_iter = 1000;

        for _ in 0..max_iter {
            let mut errors = 0;
            for i in 0..n_samples {
                let prediction = self.predict(&x[i]);
                let error = y[i] - prediction;
                if error != 0 {
                    errors += 1;
                    // Mise à jour du biais
                    self.weights[0] += learning_rate * error as f64;
                    // Mise à jour des poids des features
                    for j in 0..n_features {
                        self.weights[j + 1] += learning_rate * error as f64 * x[i][j];
                    }
                }
            }
            if errors == 0 {
                break; // convergence atteinte
            }
        }
    }

    // Prédiction par fonction signe sur le produit scalaire + biais
    fn predict(&self, x: &Self::Input) -> Self::Output {
        if self.weights.is_empty() {
            panic!("Modèle non entraîné");
        }
        let mut sum = self.weights[0];
        for i in 0..x.len() {
            sum += self.weights[i + 1] * x[i];
        }
        if sum >= 0.0 { 1 } else { -1 }
    }
}
