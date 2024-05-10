console.log("STARTING PROGRAM")

const Model = require("./model"); // PROBLEM HERE
const BatchedImageHandler = require('./image-handler-iterator')
console.log("STARTING PROGRAM 2")


async function trainModel() {
    console.log("STARTING PROGRAM")
    const model = new Model();
    console.log("CREATED MODEL INSTANCE")
    await model.compile();
    const trainingDataPath = "./training-data"
    await model.train(trainingDataPath);
}

// function testGetImages() {
//     let imageHandler = new BatchedImageHandler("./training-data", 64)
//     imageHandler.getImagePaths();
// }
//
// testGetImages()

trainModel()