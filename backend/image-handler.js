const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const faceapi = require('@vladmandic/face-api');
const sharp = require("sharp");

module.exports = class ImageHandler {
    constructor() {
        this.optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
            minConfidence: 0.4,
            maxResults: 1,
            iouThreshold: 0.3,
            scoreThreshold: 0.5,
            minFaceSize: 50,
            scoreThresholds: {}
        });
    }

    static async loadDetectionModel() {
        const modelPath = path.join(__dirname, './models');
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    }

    static async loadImages(dataPath, test = false) {
        if (!dataPath) {
            throw new Error('dataPath is required!');
        }

        if (!fs.existsSync(dataPath)) {
            throw new Error(`dataset at path "${dataPath}" doesn't exist!`);
        }

        const images = []; const labels = [];

        const datasetFolderName = test === true ? 'test' : "train";
        let datasetFolderPath = path.join(dataPath, datasetFolderName);

        if (!fs.existsSync(datasetFolderPath)) {
            datasetFolderPath = path.join(dataPath);
        }

        const labelFolders = ['real'];
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

    static async loadImageToTensor(imageBuffer) {
        const tensor = tf.node.decodeImage(imageBuffer, 3);
        return tensor;
    }

    static async cropImage(tensorImg, faceDetectionResult, index) {
        try {
            const buffer = await tf.node.encodeJpeg(tensorImg);

            const image = sharp(buffer);

            const { _x, _y, _width, _height } = faceDetectionResult._box;

            const croppedImageBuffer = await image
                .extract({
                    left: Math.floor(_x),
                    top: Math.floor(_y),
                    width: Math.floor(_width),
                    height: Math.floor(_height)
                })
                .toBuffer();

            const resizedImageBuffer = await sharp(croppedImageBuffer)
                .resize({
                    width: 200,
                    height: 200,
                    fit: 'cover',
                })
                .toBuffer();

            const outputImagePath = `./test-images/real/cropped_image${index}.jpg`;
            await fs.promises.writeFile(outputImagePath, resizedImageBuffer);

            console.log(`Cropped image saved to ${outputImagePath}`);
        } catch(error) {
            console.error(`Error cropping image ${index}: ${error.message}`);
        }
    }

    static async detectFace(imgTensor) {
        const result = await faceapi.detectSingleFace(imgTensor, this.optionsSSDMobileNet);
        return result
    }
}