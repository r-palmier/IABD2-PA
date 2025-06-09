use iabd2_pa::models::svm::SVM;
#[test]
fn test_svm_train_converges() {
    let mut model = SVM::new(2, 0.0005, 2000, 0.01, 0.5); // learning rate réduit, epochs augmentés

    let x_train = vec![
        vec![2.0, 3.0],
        vec![1.0, 1.0],
        vec![4.0, 5.0],
        vec![5.0, 2.0],
    ];
    let y_train = vec![1, 1, -1, -1];

    model.train(&x_train, &y_train);

    let mut correct = 0;
    for (x, y) in x_train.iter().zip(y_train.iter()) {
        if model.predict(x) == *y {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / y_train.len() as f64;
    assert!(accuracy >= 0.75, "Accuracy too low: {}", accuracy);
}
