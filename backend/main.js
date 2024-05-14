const Model = require("./model");
const BatchedImageHandler = require('./image-handler-iterator');
const ImageHandler = require('./image-handler');
const Utils = require('./utils');

async function trainModel() {
    console.log("STARTING PROGRAM")
    const model = new Model();
    await model.compile();
    const trainingDataPath = "./training-data"
    const numIterations = 500;

    let allImagePaths = new BatchedImageHandler(trainingDataPath).getImagePaths();
    Utils.shuffleArray(allImagePaths);

    const pathChunks = splitIntoChunks(allImagePaths, numIterations);
    console.log(pathChunks)

    for (let i = 0; i < pathChunks.length; i++) {
        console.log(`Training iteration ${i + 1}/${pathChunks.length}`);
        await model.trainAtOnce(pathChunks[i], trainingDataPath);
    }
}

function splitIntoChunks(array, numChunks) {
    const chunkSize = Math.ceil(array.length / numChunks);
    const chunks = [];
    for (let i = 0; i < array.length; i += chunkSize) {
        chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
}

trainModel()