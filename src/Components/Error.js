import React from 'react';
import error from './media/error.png';
import {Link} from 'react-router-dom';

function Error(){
    return(
        <div>
            <div style = {{display:'flex' , justifyContent:'center'}}>
                <img src = {error} alt="" width = '25%'/>
            </div>
            <div style = {{display:'flex' , justifyContent:'center', bottom:0 }}>
                <Link to = '/login'><button className = 'btn'>Click to go Back</button></Link>
            </div>
        </div>
    );
}

export default Error;