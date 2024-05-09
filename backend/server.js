const express = require('express');
const path = require('path');
const port = 8050;
const server = express();

server.use(express.json());
server.use(express.static(path.join(__dirname, '../frontend')));

server.get('/', (zahtjev, odgovor) => {
    odgovor.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

server.listen(port, () => {
    console.log(`Server pokrenut na portu: ${port}\n`);
    console.log("http://localhost:8050/");
});