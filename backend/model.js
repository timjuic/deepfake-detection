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

        this.model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
        this.model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

        this.model.add(tf.layers.flatten());

        this.model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        this.model.add(tf.layers.dropout({ rate: 0.2 }));
        this.model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

        const optimizer = tf.train.adam(0.00005)
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


        const uniqueLabels = [...new Set(labels)];
        const ys = tf.oneHot(tf.tensor1d(labels.map(label => uniqueLabels.indexOf(label)), 'int32'), uniqueLabels.length);
        const xs = tf.stack(images);

        console.log("Started training model...");
        await this.model.fit(xs, ys, {
            epochs: 5,
            validationSplit: 0.1,
            batchSize: 32,
            verbose: 2,
            shuffle: true,
            callbacks: [
                tf.callbacks.earlyStopping({ monitor: 'loss', patience: 5 }),
                tf.node.tensorBoard('logs'),
            ],
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