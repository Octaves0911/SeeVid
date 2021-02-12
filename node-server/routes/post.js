const router = require('express').Router();
const User = require('../models/User.js');
const verify = require('./verification.js');
const jwt = require('jsonwebtoken');
const Video = require('../models/Video.js');
const Axios = require('axios');

router.get('/dashboard',verify,async (req,res)=>{
    //post api
    const token = req.header('Cookie').split(';')[0].split('=')[1];
    try{
        const username = jwt.decode(token).username;
        const user = await User.findOne({username : username});
        var response = new Object();
        response.message = 'Library';
        response.code = 0;
        response.values = user.Videos.library;
        response.user = {name: user.username , email : user.email};
        res.status(200).send(response);
    }
    catch(err){
        const response = new Object();
        response.code = 1;
        response.message = 'Token exception';
        res.status(403).send(response);
    }
})

router.post('/upload' , verify , async (req,res)=>{
    const token = req.header('Cookie').split(';')[0].split('=')[1];
    const username = jwt.decode(token).username;

    const user = await User.findOne({username : username});
    try{
        const flask_payload = {
            "url" : req.body.url,
            "Query" : req.body.Query
        }
        
        console.log('sending request');
        await Axios.post('http://b5bfda7e988e.ngrok.io/videourl' , flask_payload).then(async (resp)=>{
            // console.log(typeof(res.data.timestamp));
            // let flask_response = new Object();
            // flask_response.name = req.body.url.split('&')[1].split('=')[1];
            // flask_response.url = req.body.url;
            // flask_response.description = new Array();
            // flask_response.description = resp.data.description

            // Video.library.push(flask_response);
            // await Video.save();

            let request = new Object();
            request.name = req.body.VideoName;
            request.url = req.body.url;
            request.query = req.body.Query;
            request.timestamp = new Array();
            // console.log(Object.keys(res.data.timestamp[0]));
            request.timestamp = resp.data.timestamp;
            user.Videos.library.push(request);
            await user.save();
            const lib = user.Videos.library;
            let obj = new Object();
            obj = lib[lib.length - 1];
            
            const response = new Object();
            response.vid = obj;
            response.code = 0;
            response.message = 'Video Uploaded Successfully. Please wait, you will be redirected shortly.';
            console.log('model finished');
            res.status(200).send(response);
        }).catch((err)=>{
            console.log(err);
            const response = new Object();
            response.code = 1;
            response.message = 'Database Exception';
            res.header('Content-Type','application/json')
            res.status(500).send(response);
        });
    }
    catch{
        console.log('error');
        const response = new Object();
        response.code = 1;
        response.message = 'Database Exception';
        res.header('Content-Type','application/json')
        res.status(500).send(response);
    }

})

router.post('/player', verify , async (req,res)=>{
    const token = req.header('Cookie').split(';')[0].split('=')[1];
    const username = jwt.decode(token).username;
    const user = await User.findOne({username : username});
    try{
        const lib = user.Videos.library;

        let obj = new Object();
        let arr = [{}];
        for(let i = 0 ; i < lib.length ; ++i ){
            if(lib[i]._id == req.body.link){
                obj = lib[i];
            }
            else
                arr.push(lib[i]);
        }
        if(Object.keys(obj).length === 0){
            const response = new Object();
            response.code = 0;
            response.message = 'Video not found in library';
            res.status(400).send(response);
        }
        else{
            const response =new Object();
            response.results = obj;
            response.code = 0;
            response.playlists = arr;
            res.status(200).send(response);
        }
    }
    catch(e){
        const response = new Object();
        response.code = 0;
        response.message = 'Database Exception';
        res.status(500).send(response);
    }
})

module.exports = router;