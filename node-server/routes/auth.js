const router = require('express').Router();
const bcrypt = require('bcrypt');
const User = require('../models/User.js');
const {RegisterValidations,LoginValidations} = require('../validations.js');
const jwt = require('jsonwebtoken');

router.post('/register',async (req,res)=>{
    //register auth
    RegisterValidations(req.body).then(async(result)=>{
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(req.body.password , salt);

        const user = new User({
            username: req.body.username,
            email : req.body.email,
            password : hashedPassword,
            processing : false
        })
        try{
            const savedUser = await user.save();
            const token = jwt.sign({username : user.username , email : user.email} , process.env.TOKEN_SECRET);
            res.cookie('token' , token);
            res.header("Content-Type", "application/json");
            res.status(200);
            const response = new Object();
            response.message = 'Registered Successfully';
            response.code = 0;
            res.send(response);
        }
        catch(err){
            var response = new Object();
            response.error = 'Database Exception';
            response.message = err.message;
            response.code = 1;
            res.header("Content-Type", "application/json")
            res.status(500).send(response); 
        }

    }).catch((err)=>{
        var response = new Object();
        response.error = "ValidationException";
        response.message = err.message;
        res.header("Content-Type", "application/json")
        res.status(400).send(response);
    })
})

router.post('/login' , async (req,res)=>{
    LoginValidations(req.body).then(async (result)=>{
        var token = null;
        const user = await User.findOne({email : req.body.email});
        token = jwt.sign({username : user.username , email : user.email} , process.env.TOKEN_SECRET);
        res.cookie('token',token);
        res.header('Content-Type','application/json');
        var response = new Object();
        response.message = 'Logged In';
        response.code = 0;
        response.username = user.username;
        res.status(200).send(response);
    }).catch((err)=>{
        var response = new Object();
        response.error = 'Validation Error';
        response.message = err.message;
        res.header("Content-Type","application/json");
        res.status(400).send(response);
    })
})

module.exports = router;