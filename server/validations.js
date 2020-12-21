const Joi = require('@hapi/joi');
const User = require('./models/User.js'); 
const bcrypt = require('bcrypt');

const RegisterValidations = (data) =>{
    const RegisterSchema = Joi.object({
        username: Joi.string()
            .min(6)
            .required(),
        email: Joi.string()
            .required()
            .email(),
        password: Joi.string()
            .min(6)
            .required(),
        
    });
    return new Promise(async function(resolve, reject) {
        var validatedObject = RegisterSchema.validate(data);
        if(!validatedObject.error) {
            try{
                const emailexists = await User.findOne({email : data.email});
                try{
                    const username_exists = await User.findOne({name : data.username});
                    resolve("Validation Successful");
                }
                catch(err){
                    reject(new Error("Username exists"));
                }
            }
            catch(err){
                reject(new Error("Email exists"));
            }
        }
        else {
            reject(new Error(validatedObject.error.details[0].message));
        }
    });
}

const loginValidation = (data) => {
    const loginSchema = Joi.object({
        email: Joi.string()
            .required()
            .email(),
        password: Joi.string()
            .min(6)
            .required()
    });
    return new Promise(async (resolve , reject)=>{
        const validatedObject2 = loginSchema.validate(data);
        if(!validatedObject2.error){
            try{
                const user = await User.findOne({email : data.email});
                const passexists = await bcrypt.compare(data.password ,user.password);
                resolve("Validation Successful");
            }
            catch(err){
                reject(new Error('Email or Password is Wrong'))
            }
        }
        else{
            reject(new Error(validatedObject2.error.details[0].message));
        }
    })
}
module.exports.RegisterValidations = RegisterValidations;
module.exports.LoginValidations = loginValidation;
