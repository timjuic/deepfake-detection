const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');

class ImageHandler {
    constructor() {
    }

    async loadImages(dataPath, test) {
        if (!dataPath) {
            throw new Error('dataPath is required!');
        }

        if (!fs.existsSync(dataPath)) {
            throw new Error(`dataset at path "${dataPath}" doesn't exist!`);
        }

        console.log(`Loading images from ${dataPath}`);

        const images = [];
        const labels = [];

        const datasetFolderName = test === true ? 'test' : "train";
        let datasetFolderPath = path.join(dataPath, datasetFolderName);

        if (!fs.existsSync(datasetFolderPath)) {
            datasetFolderPath = path.join(dataPath);
        }

        const labelFolders = ['real', 'deepfake'];
        let startTime = Date.now();
        for (const labelFolder of labelFolders) {
            const labelFolderPath = path.join(datasetFolderPath, labelFolder);

            if (!fs.existsSync(labelFolderPath)) {
                throw new Error(`Folder for label "${labelFolder}" doesn't exist! Check the dataset!`)
                continue;
            }

            const imageFiles = fs.readdirSync(labelFolderPath);
            console.log(`Started loading ${labelFolderPath}`)

            for (const imageFile of imageFiles) {
                const imagePath = path.join(labelFolderPath, imageFile);

                try {
                    const imageBuffer = fs.readFileSync(imagePath);
                    const tensorImage = await this.loadImageToTensor(imageBuffer);
                    images.push(tensorImage);
                    labels.push(labelFolder); // Use folder name as label
                } catch (error) {
                    console.warn(`Skipping ${imagePath} - Error: ${error.message}`);
                }
            }
        }

        let endTime = Date.now();
        console.log(`Loaded ${images.length} images in ${endTime - startTime} ms`);

        const uniqueLabels = [...new Set(labels)];
        console.log("Unique Labels:", uniqueLabels);
        console.log("Label Indices:", uniqueLabels.map((label, index) => `${label}: ${index}`));

        return { images, labels };
    }

    async loadImageToTensor(imageBuffer) {
        const tensor = tf.node.decodeImage(imageBuffer, 3);
        return tensor;
    }
}

let imageHandler = new ImageHandler()
imageHandler.loadImages("./datasets/dataset-4", false);