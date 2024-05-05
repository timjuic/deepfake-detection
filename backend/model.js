const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const sharp = require("sharp");

class Model {
    constructor() {
        this.model = tf.sequential();
    }
    async compile() {
        this.model.add(tf.layers.inputLayer({ inputShape: [100, 100, 3] }));

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
    }

    async train() {
        const { images, labels } = await this.loadAndPreprocessData(path.join(__dirname, 'data', 'affectnet-dataset', 'train'));

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

        await this.model.save('file://trained-model-raf');
        console.log("Model saved");
    }

    async load(filepath) {
        this.model = await tf.loadLayersModel(filepath);
    }

    async loadAndPreprocessData(dataPath) {
        console.log(`Loading and preprocessing data at ${dataPath}`);

        const csvFilePath = path.join(dataPath, 'labels.csv');
        const csvStream = fs.createReadStream(csvFilePath).pipe(csv());

        const images = [];
        const labels = [];
        let numberOfAugmentations = 4

        for await (const entry of csvStream) {
            if (entry.label === 'contempt') {
                continue;
            }

            const imagePath = path.join(dataPath, entry.pth);

            try {
                const imageBuffer = fs.readFileSync(imagePath);
                const originalImage = tf.node.decodeImage(imageBuffer, 3);

                images.push(originalImage);
                labels.push(entry.label);

                await this.saveImageToFile(imagePath, "./testimg", originalImage);

                for (let i = 0; i < numberOfAugmentations; i++) {
                    let augmentedImage = await this.augmentImage(imageBuffer, i);
                    images.push(augmentedImage);
                    labels.push(entry.label);
                    let imageName = imagePath.split('.').map((part, partNum) => {
                        if (partNum === 0) part += `-${i}`
                        return part;
                    }).join('.')
                    await this.saveImageToFile(imageName, "./testimg", augmentedImage)
                }
            } catch (error) {
                console.warn(`Skipping ${imagePath} - Error: ${error.message}`);
            }
        }

        console.log(`Finished loading data`);

        const uniqueLabels = [...new Set(labels)];
        console.log("Unique Labels:", uniqueLabels);
        console.log("Label Indices:", uniqueLabels.map((label, index) => `${label}: ${index}`));

        return { images, labels };
    }

    async augmentImage(imageBuffer, callCount) {
        const buffer = await sharp(imageBuffer).toBuffer();

        const tensor = tf.node.decodeImage(buffer, 3);

        switch (callCount) {
            case 0:
                const flippedBuffer = await sharp(buffer).flop().toBuffer();
                return tf.node.decodeImage(flippedBuffer, 3);

            case 1:
                const adjustedBuffer = await sharp(buffer).modulate({ brightness: 1.1 }).toBuffer();
                return tf.node.decodeImage(adjustedBuffer, 3);

            case 2:
                const adjustedBuffer2 = await sharp(buffer).modulate({ brightness: 0.9 }).toBuffer();
                return tf.node.decodeImage(adjustedBuffer2, 3);
            case 3:
                const targetSize = 100;
                const fixedZoomFactor = 1.1;

                const newWidth = Math.floor(targetSize * fixedZoomFactor);
                const newHeight = Math.floor(targetSize * fixedZoomFactor);

                const zoomedBuffer = await sharp(buffer).resize({
                    width: newWidth,
                    height: newHeight,
                }).toBuffer();

                const croppedBuffer = await sharp(zoomedBuffer).extract({
                    left: Math.floor((newWidth - targetSize) / 2),
                    top: Math.floor((newHeight - targetSize) / 2),
                    width: targetSize,
                    height: targetSize,
                }).toBuffer();

                return tf.node.decodeImage(croppedBuffer, 3);

            default:
                return tensor;
        }
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

// Usage:
const model = new Model();
model.train(); // To train the model
// model.load('file://trained-model-raf'); // To load the trained model from file
