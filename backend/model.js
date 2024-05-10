const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const sharp = require("sharp");
const ImageHandler = require("./image-handler");
const BatchedImageHandler = require('./image-handler-iterator');

module.exports = class Model {
    constructor() {
        this.model = tf.sequential();
    }
    async compile() {
        console.log("COMPILE STARTED")
        this.model.add(tf.layers.inputLayer({ inputShape: [200, 200, 3] }));

        this.model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

        this.model.add(tf.layers.flatten());

        this.model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        this.model.add(tf.layers.dropout({ rate: 0.4 }));
        this.model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

        const optimizer = tf.train.adam(0.00005)
        this.model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        console.log("COMPILE FINISHED")
    }

    async train(pathToTrainingData) {
        // Create a dataset using tf.data.generator
        let batchedImageHandler = new BatchedImageHandler(pathToTrainingData, 128);
        let dataset = tf.data.generator(() => batchedImageHandler.loadImageGenerator());
        dataset = dataset.shuffle(500);
        dataset = dataset.batch(128);

        console.log("Started training model...");
        await this.model.fitDataset(dataset, {
            epochs: 25,
            verbose: 2,
            callbacks: [
                tf.callbacks.earlyStopping({ monitor: 'loss', patience: 5 }),
                tf.node.tensorBoard('logs')
            ]
        });

        await this.model.save('file://trained-model');
        console.log("Model saved");
    }

    async load(filepath) {
        this.model = await tf.loadLayersModel(filepath);
    }


    async saveImageToFile(imagePath, outputFolder, normalizedFaceImage) {
        const imageName = path.basename(imagePath);
        const outputImagePath = path.join(outputFolder, imageName);
        tf.setBackend("tensorflow");

        try {
            const faceImageBuffer = Buffer.from((await tf.node.encodeJpeg(normalizedFaceImage)).buffer);
            await sharp(faceImageBuffer).toFile(outputImagePath);
            console.log(`Image saved to ${outputImagePath}`);
        } catch (error) {
            console.error(`Error saving image to ${outputImagePath}: ${error.message}`);
        }
    }
}