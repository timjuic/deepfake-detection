const tf = require("@tensorflow/tfjs-node");
const path = require('path');
const fs = require('fs');
const Utils = require('./utils');

module.exports = class BatchedImageHandler {
    constructor(dataPath, batchSize) {
        this.dataPath = dataPath;
        this.batchSize = batchSize;
        this.labelFolders = ['real', 'deepfake'];
    }

    async *loadImageGenerator() {
        const allImagePaths = this.getImagePaths();
        Utils.shuffleArray(allImagePaths);

        let imageIndex = 0;
        while (imageIndex < allImagePaths.length) {
            console.log(`current img: ${allImagePaths[imageIndex]}`);
            const batchImagePaths = allImagePaths.slice(imageIndex, imageIndex + this.batchSize);

            // Load, process, and yield the batch
            const xs = tf.stack(await Promise.all(batchImagePaths.map(async imagePath => {
                const imageBuffer = fs.readFileSync(imagePath);
                return this.loadImageToTensor(imageBuffer);
            })));

            const ys = tf.oneHot(batchImagePaths.map(imagePath => {
                // Derive label from 'real' or 'deepfake' in path
                const folder = path.basename(path.dirname(imagePath));
                return this.labelFolders.indexOf(folder);
            }), 2); // Assuming two labels

            yield { xs, ys };

            imageIndex += this.batchSize;
        }
    }

    getImagePaths() {
        const allowedExtensions = ['.jpg', '.jpeg', '.png']; // Allowed file extensions
        const imagePaths = [];

        for (const labelFolder of this.labelFolders) {
            const folderPath = path.join(this.dataPath, labelFolder);
            const imageFiles = fs.readdirSync(folderPath);

            const filteredImageFiles = imageFiles.filter(imageFile => {
                const ext = path.extname(imageFile).toLowerCase();
                return allowedExtensions.includes(ext);
            });

            imagePaths.push(...filteredImageFiles.map(imageFile => path.join(folderPath, imageFile)));
        }

        return imagePaths;
    }

    async loadImageToTensor(imageBuffer) {
        const tensor = tf.node.decodeImage(imageBuffer, 3);
        return tensor;
    }
}