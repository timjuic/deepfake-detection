const sharp = require("sharp");
const tf = require("@tensorflow/tfjs-node-gpu");

module.exports = class ImageAugmentor {
    static async augmentImage(imageBuffer, callCount) {
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
}