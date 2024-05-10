const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const sharp = require("sharp");
const {ImageHandler} = require("./image-handler");

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
        const { images, labels } = await ImageHandler.loadImages(pathToTrainingData)
        // const { images, labels } = await this.loadAndPreprocessData(path.join(__dirname, 'data', 'affectnet-dataset', 'train'));

        const uniqueLabels = [...new Set(labels)];
        const ys = tf.oneHot(tf.tensor1d(labels.map(label => uniqueLabels.indexOf(label)), 'int32'), uniqueLabels.length);
        const xs = tf.stack(images);

        console.log("Started training model...");
        await this.model.fit(xs, ys, {
            epochs: 25,
            validationSplit: 0.1,
            batchSize: 128,
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