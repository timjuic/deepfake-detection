const express = require('express');
const multer = require('multer');
const path = require('path');
const port = 8050;
const server = express();

const ImageHandler = require('./image-handler');
const Model = require('./model');
const upload = multer({ storage: multer.memoryStorage() });

server.use(express.json());
server.use(express.static(path.join(__dirname, '../frontend')));

server.get('/', (zahtjev, odgovor) => {
    odgovor.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

server.post('/upload-image', upload.single('image'),  async (zahtjev, odgovor) =>{
    try {
          const slikaTensor = await ImageHandler.loadImageToTensor(zahtjev.file.buffer);
          await ImageHandler.loadDetectionModel();
          const detektirajLice = await ImageHandler.detectFace(slikaTensor);
          if(detektirajLice != undefined){
                var cropped = await ImageHandler.cropImage(slikaTensor, detektirajLice);
                
                const model = new Model();
                await model.load('file://trained-model/model.json');

                var result = await model.predict(cropped);
            
                if(result == 'real'){
                    odgovor.status(200).json({
                        success: true,
                        message: 'Real' 
                    });
                } else {
                    odgovor.status(200).json({
                        success: true,
                        message: 'DeepFake' 
                    });
                }
          }else{
            odgovor.status(400).json({
                success: false,
                message: 'Lice nije pronađeno!' }
            );
          }
    } catch (error) {
        odgovor.status(500).json({
            success: false,
            message: 'Error' + error }
        );
    }
});

server.listen(port, () => {
    console.log(`Server pokrenut na portu: ${port}\n`);
    console.log("http://localhost:8050/");
});