const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const sharp = require("sharp");
const BatchedImageHandler = require('./image-handler-iterator');
const ImageHandler = require('./image-handler');


module.exports = class Model {
    constructor() {
        this.model = tf.sequential();
    }


    async compile() {
        await tf.setBackend('tensorflow');

        this.model.add(tf.layers.inputLayer({ inputShape: [200, 200, 3] }));

        // Convolutional Layers with Batch Normalization
        this.model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', padding: 'same', kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

        this.model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same', kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

        // Slightly Deeper Network with 128 Filters
        this.model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same', kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

        // Flatten and Dense Layers
        this.model.add(tf.layers.flatten());
        this.model.add(tf.layers.dense({ units: 256, activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }));
        this.model.add(tf.layers.dropout({ rate: 0.5 })); // Increased dropout for stronger regularization
        this.model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

        // Learning Rate and Optimizer
        const optimizer = tf.train.adam(0.0001); // Reduced learning rate

        this.model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    }


    async trainBatched(pathToTrainingData) {
        let batchedImageHandler = new BatchedImageHandler(pathToTrainingData, 64);
        let dataset = tf.data.generator(() => batchedImageHandler.loadImageGenerator());
        dataset = dataset.shuffle(100);
        dataset = dataset.batch(64);


        console.log("Started training model...");
        await this.model.fitDataset(dataset, {
            epochs: 25,
            verbose: 2,
            yieldEvery: 'batch',
            callbacks: [
                tf.node.tensorBoard('logs'),
            ]
        });

        await this.model.save('file://trained-model');
        console.log("Model saved");
    }

    async trainAtOnce(imagePaths) {
        const labelFolders = ['real', 'deepfake'];
        const { images, labels } = await ImageHandler.loadImagesFromChunks(imagePaths, labelFolders);

        const uniqueLabels = [...new Set(labels)];
        const ys = tf.oneHot(tf.tensor1d(labels.map(label => uniqueLabels.indexOf(label)), 'int32'), uniqueLabels.length);
        const xs = tf.stack(images);

        console.log("Started training model...");
        await this.model.fit(xs, ys, {
            epochs: 10,
            validationSplit: 0.2,
            batchSize: 16,
            verbose: 2,
            shuffle: true,
            callbacks: [
                tf.callbacks.earlyStopping({ monitor: 'loss', patience: 3 }),
                tf.node.tensorBoard('logs'),
            ],
        });
        await this.model.save('file://trained-model');

        console.log("Model saved");
    }

    async load(filepath) {
        this.model = await tf.loadLayersModel(filepath);
    }


    async predict(tensorImage) {
        tensorImage = tensorImage.expandDims(0);

        const predictions = this.model.predict(tensorImage);

        const predictedClassIndex = predictions.argMax(1).dataSync()[0];
        const predictedLabel = ['real', 'deepfake'][predictedClassIndex];

        return predictedLabel;
    }
}