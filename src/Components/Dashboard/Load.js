import React  from 'react';
import './Dashboard.css';
import CircularProgress from '@material-ui/core/CircularProgress';

function Load(){
    return(
        <div>
            <div style = {{display: 'grid' , placeItems:'center' , marginTop:'5%',marginBottom:'8%'}}><CircularProgress/><h1>Loading Content</h1></div>
        </div>
    );
}

export default Load;
