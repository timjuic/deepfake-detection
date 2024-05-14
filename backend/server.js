const express = require('express');
const path = require('path');
const port = 8050;
const server = express();

const model = require('./model');

server.use(express.json());
server.use(express.static(path.join(__dirname, '../frontend')));

server.get('/', (zahtjev, odgovor) => {
    odgovor.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

server.post('/upload-image', async (zahtjev, odgovor) =>{
    try {
          const slikaTensor = await ImageHandler.loadImageToTensor(zahtjev.body.image);
          const detektirajLice = await ImageHandler.detectFace(slikaTensor);
      
          if(detektirajLice != undefined){
                await ImageHandler.cropImage(slikaTensor, detektirajLice);
                await model.load('./trained-model');

                var result = await model.predict(slikaTensor);
            
                if(result == 'real'){
                    odgovor.status(200).json({
                        success: true,
                        message: 'Real' 
                    });
                }

                odgovor.status(200).json({
                    success: true,
                    message: 'DeepFake' 
                });
          }
      
          odgovor.status(400).json({
              success: false,
              message: 'Lice nije pronađeno!' }
          );
    } catch (error) {
        odgovor.status(500).json({
            success: false,
            message: 'Error' }
        );
    }
});

server.listen(port, () => {
    console.log(`Server pokrenut na portu: ${port}\n`);
    console.log("http://localhost:8050/");
});