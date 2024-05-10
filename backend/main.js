console.log("STARTING PROGRAM")

const Model = require("./model"); // PROBLEM HERE
console.log("STARTING PROGRAM 2")


async function trainModel(pathToTrainingData) {
    console.log("STARTING PROGRAM")
    const model = new Model();
    console.log("CREATED MODEL INSTANCE")
    await model.compile();
    const trainingDataPath = "./training-data"
    await model.train(pathToTrainingData);
}

trainModel()