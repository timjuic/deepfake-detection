const ImageHandler = require("./image-handler");


async function startHandler() {
    let data = await ImageHandler.loadImages("./datasets/dataset-1", false);
    ImageHandler.loadDetectionModel()
        .then(async () => {
            let counter = 54466;
            for (let image of data.images) {
                console.log("for each")
                let detectionResult = await ImageHandler.detectFace(image);
                if (detectionResult === undefined) continue;
                await ImageHandler.cropImage(image, detectionResult, counter)
                counter++;
            }
        })
}

startHandler()