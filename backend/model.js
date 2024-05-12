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
        console.log("COMPILE STARTED")
        await tf.setBackend('tensorflow');
        this.model.add(tf.layers.inputLayer({ inputShape: [200, 200, 3] }));

        this.model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

        this.model.add(tf.layers.flatten());

        this.model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        this.model.add(tf.layers.dropout({ rate: 0.2 }));
        this.model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

        const optimizer = tf.train.adam(0.001)
        this.model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        console.log("COMPILE FINISHED")
    }

    async trainBatched(pathToTrainingData) {
        // Create a dataset using tf.data.generator
        let batchedImageHandler = new BatchedImageHandler(pathToTrainingData, 64);
        let dataset = tf.data.generator(() => batchedImageHandler.loadImageGenerator());
        // const ds = tf.data.zip(dataset).shuffle(100).batch(32)
        dataset = dataset.shuffle(100);
        dataset = dataset.batch(64);


        console.log("Started training model...");
        await this.model.fitDataset(dataset, {
            epochs: 25,
            verbose: 2,
            yieldEvery: 'batch',
            callbacks: [
                // tf.callbacks.earlyStopping({ monitor: 'loss', patience: 5 }),
                tf.node.tensorBoard('logs'),

            ]
        });

        await this.model.save('file://trained-model');
        console.log("Model saved");
    }

    async trainAtOnce(imagePathChunks) {
        const labelFolders = ['real', 'deepfake'];
        const { images, labels } = await ImageHandler.loadImagesFromChunks(imagePathChunks, labelFolders);


        let trainModel = async () => {
            const uniqueLabels = [...new Set(labels)];
            const ys = tf.oneHot(tf.tensor1d(labels.map(label => uniqueLabels.indexOf(label)), 'int32'), uniqueLabels.length);
            const xs = tf.stack(images);

            console.log("Started training model...");
            await this.model.fit(xs, ys, {
                epochs: 5,
                validationSplit: 0.2,
                batchSize: 16,
                verbose: 2,
                shuffle: true,
                callbacks: [
                    tf.callbacks.earlyStopping({ monitor: 'loss', patience: 5 }),
                    tf.node.tensorBoard('logs'),
                ],
            });
            await this.model.save('file://trained-model');
        }

        await trainModel();
        await tf.disposeVariables()
        images.forEach(image => image.dispose());
        labels.length = 0;

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