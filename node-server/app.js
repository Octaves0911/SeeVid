const express = require('express');
const app = express();
const env = require('dotenv');
const cors = require('cors');
const cookieparser = require('cookie-parser');

//importing routes
const authRoutes = require('./routes/auth.js');
const postRoutes = require('./routes/post.js');
const mongoose = require('mongoose');

//config dotenv file
env.config();

//connect to db
mongoose.connect(process.env.DB_CONNECT ,{ useUnifiedTopology: true ,useNewUrlParser: true }, (err)=>{
    if(err){
        console.log(err.message);
    }
    else{
        console.log('Connected to db');
    }
} )

//middleware
app.use(cookieparser());
app.use(cors({origin:"http://localhost:3000",credentials:true}));
app.use(express.json());

// Defining Routes
app.use('/api/user/',postRoutes);
app.use('/api/user/',authRoutes);


// listening to port

app.listen(process.env.PORT , ()=>{console.log('Listening on port 8080..')});

