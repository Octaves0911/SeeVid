const boolean = require('@hapi/joi/lib/types/boolean');
const mongoose = require('mongoose')

const video = new mongoose.Schema({
    
    Video : {
        library : [{
            url : String,
            name : String,
            description : [[]]
        }]
    },
})

module.exports = mongoose.model('Video',video);