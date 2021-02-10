const { type } = require('@hapi/joi/lib/extend');
const mongoose = require('mongoose')

const user = new mongoose.Schema({
    username : {
        type: String,
        required : true,
        min: 5,
        max: 20,

    },
    email : {
        type : String,
        required : true,
        min : 5,
        max : 255,
    },
    password : {
        type : String,
        required : true,
        min : 6,
        max : 1024
    },
    Videos : {
        library : [{
            name : String,
            url : String,
            query : String,
            timestamp : [[]]
        }]
    },
})

module.exports = mongoose.model('User',user);