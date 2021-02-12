const jwt = require('jsonwebtoken');

module.exports = function(req,res,next){
    if(!req.header('Cookie')) {
        const response = new Object();
        response.message = "Access Denied";
        response.code = "1";
        res.status(401);
        res.send(response);
    }
    else {
    const token = req.header('Cookie').split(';')[0].split('=')[1];
    if(!token){
        const response = new Object();
        response.message = "Access Denied";
        response.code = "1";
        res.status(401);
        res.send(response);
    }
    try{
        const verified = jwt.verify(token, process.env.TOKEN_SECRET);
        
        req.user = verified;
        next();
    }
    catch(err){
        const response = new Object();
        response.message = "Invalid Token";
        response.code = "1";
        res.status(400);
        res.send(response);
    }
}
}