// Import TensorFlow.js
const tf = require('@tensorflow/tfjs');

// Generate random data for training
const numSamples = 1000000;
const trueCoefficients = { a: 2, b: 1 }; // True coefficients for y = ax + b + noise
const noise = tf.randomNormal([numSamples], 0, 0.5); // Noise for simulating real data

function generateData(numSamples, coefficients, noise) {
    const xs = tf.randomUniform([numSamples], 0, 1); // Random x values between 0 and 1
    const ys = xs.mul(coefficients.a).add(coefficients.b).add(noise);
    return { xs, ys };
}

const data = generateData(numSamples, trueCoefficients, noise);

// Define and compile the model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] })); // Single dense layer for linear regression
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Training the model
const { xs, ys } = data;
model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch + 1}: Loss: ${logs.loss}`);
        }
    }
}).then(info => {
    console.log('Final accuracy', info.history.loss);
    // Use the trained model for prediction
    const testX = tf.tensor2d([0.1, 0.2, 0.3], [3, 1]); // Test data for prediction
    const predictions = model.predict(testX);
    predictions.print();
});
