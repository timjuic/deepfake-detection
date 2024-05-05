const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const faceapi = require('@vladmandic/face-api');
const sharp = require("sharp");

class ImageHandler {
    constructor() {
        this.optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
            minConfidence: 0.4, // Decreased min confidence threshold to make detection more sensitive
            maxResults: 1, // Increased max results to detect more faces per image
            iouThreshold: 0.3, // Default value for IoU threshold
            scoreThreshold: 0.5, // Decreased score threshold to include more detections with lower confidence
            minFaceSize: 50, // Decreased min face size to detect smaller faces
            scoreThresholds: {} // Default score thresholds for different face classes
        });
    }

    async loadDetectionModel() {
        const modelPath = path.join(__dirname, './models');
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    }

    async loadImages(dataPath, test) {
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

    async loadImageToTensor(imageBuffer) {
        const tensor = tf.node.decodeImage(imageBuffer, 3);
        return tensor;
    }

    async cropImage(tensorImg, faceDetectionResult, index) {
        try {
            // Load the original image
            const buffer = await tf.node.encodeJpeg(tensorImg);

            // Load the original image buffer
            const image = sharp(buffer);

            // Get the bounding box coordinates
            const { _x, _y, _width, _height } = faceDetectionResult._box;

            // Crop the image based on the bounding box
            const croppedImageBuffer = await image
                .extract({
                    left: Math.floor(_x),
                    top: Math.floor(_y),
                    width: Math.floor(_width),
                    height: Math.floor(_height)
                })
                .toBuffer();

            // Resize the cropped image to the target size
            const resizedImageBuffer = await sharp(croppedImageBuffer)
                .resize({
                    width: 200,
                    height: 200,
                    fit: 'cover', // Cover mode ensures the image completely fills the target size without black margins
                })
                .toBuffer();

            // Save the cropped image to the output folder
            const outputImagePath = `./test-images/real/cropped_image${index}.jpg`;
            await fs.promises.writeFile(outputImagePath, resizedImageBuffer);

            console.log(`Cropped image saved to ${outputImagePath}`);
        } catch(error) {
            console.error(`Error cropping image ${index}: ${error.message}`);
        }
    }

    async detectFace(imgTensor) {
        const result = await faceapi.detectSingleFace(imgTensor, this.optionsSSDMobileNet);
        return result
    }
}

async function startHandler() {
    let imageHandler = new ImageHandler()
    let data = await imageHandler.loadImages("./datasets/dataset-1", false);
    imageHandler.loadDetectionModel()
        .then(async () => {
            let counter = 54466;
            for (let image of data.images) {
                console.log("for each")
                let detectionResult = await imageHandler.detectFace(image);
                if (detectionResult === undefined) continue;
                await imageHandler.cropImage(image, detectionResult, counter)
                counter++;
            }
        })
}

startHandler()