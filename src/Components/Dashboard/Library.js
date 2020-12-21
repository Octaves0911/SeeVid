import React, {useEffect, useState} from 'react';
import './Dashboard.css';
import Axios from 'axios';
import RenderCards from './RenderCards.js';
import ErrorOutlineIcon from '@material-ui/icons/ErrorOutline';
import Load from './Load.js';
import urls from '../../config.js';


function Library (){
    const [Videos , setVideos] = useState([]);
    const [Data, setData] = useState(false);
    useEffect(()=>{
        const url1 = `${urls.node_url}dashboard`;
        Axios.get(url1 , {withCredentials : true})
        .then((res)=>{
            setVideos(res.data.values);
            setData(true);
        })
        .catch((err)=>{
            setData(true);

        })
    },[])
    const vids = Videos.map((vid)=>{
        let temp = vid.url.split('&')[0];
        return(
                <div className = 'col-12 col-sm-3 mb-4 mt-4'>
                    <RenderCards url = {temp} name = {vid.name} query = {vid.query} id = {vid._id}/>
                </div>
        );
    })
    const library = ()=>{
        if(Videos.length > 0){
            return(
                vids
            );
        }
        else{
            return(
                <div style = {{marginTop : '5%' , marginBottom : '8%'}}><h1 style = {{textAlign:'center'}}><ErrorOutlineIcon fontSize = 'large'/> Upload Videos to add them to your Library </h1></div>
            );
        }
    }
    return(
        <div className = 'library container mt-5'>
            <div className = 'row justify-content-center align-items-center'>
                {Data? library() : <Load/>}
             
            </div>
        </div>
    );
}

export default Library;