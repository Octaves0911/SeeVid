import React, { useState , useEffect} from 'react';
import './Dashboard.css';
import Header from './Header.js';
import CloseIcon from '@material-ui/icons/Close';
import {TextField} from '@material-ui/core';
import {Button} from '@material-ui/core';
import {FormControlLabel,Checkbox} from '@material-ui/core';
import Axios from 'axios';
import {Redirect} from 'react-router-dom';
import MuiAlert from '@material-ui/lab/Alert';
import urlParser from "js-video-url-parser";
import urls from '../../config.js';


function Alert(props) {
    return <MuiAlert elevation={6} variant="filled" {...props} />;
  }

//   const useStyles = makeStyles((theme) => ({
//     root: {

//       width: '25%',
//       '& > * + *': {
//         marginTop: theme.spacing(1),
//       },
//     },
//   }));

function Upload() {
    // const classes = useStyles();
    const [Check , setCheck] = useState(false);
    const [Url , setUrl] = useState('');
    const [Name , setName] = useState('');
    const [Log , setLog] = useState(false);
    const [Err , setErr] = useState('');
    const [Error , setError] = useState(false);
    const [Upload , setUpload] = useState(false);
    const [Query , setQuery] = useState('');
    const [Open, setOpen] = useState(false);
    const [RedirectUrl , setRedirectUrl] = useState('');
    const [ Dash , setDash] = useState(false);
      useEffect(()=>{
          Axios.get(`${urls.node_url}dashboard`,{withCredentials : true}).then((res)=>{
              setLog(false);
            }).catch((err)=>{
                setLog(true);
            })
            Axios.get(`${urls.flask_url}videourl`).then((res)=>{
                console.log(res);
            }).catch((err)=>{
                setDash(true);
            })
        },[]);
    
    const handleClick = ()=>{

        if(Url === '' || Name === '' || Query === ''){
            setErr('Please Enter all the fields');
            setError(true);
        }
        else if(!checkUrl()){
            setError(true);
        }
        else{
                setOpen(true);
                setError(false);
                var payload = {
                    "url" : Url,
                    "VideoName" : Name,
                    "Query" : Query
                };
                const url = `${urls.node_url}upload`;
                Axios.post(url , payload , {
                    "headers":{
                        "Accept": 'application/json',
                        "content-type":"application/json",
                    },
                    withCredentials : true
                }).then(async (res)=>{
                    const link = `/player/q=${res.data.vid.name}&id=${res.data.vid._id}`;
                    setRedirectUrl(link);
                    setUpload(true);
                    const payload_flask = {
                        "url" : Url,
                        "Query" : Query
                    }
                    let flask_url = `${urls.flask_url}videourl`;
                    await Axios.post(flask_url , payload_flask , {
                        "headers":{
                            "Accept": 'application/json',
                            "content-type":"application/json",
                        }
                    }).then((res)=>{
                        console.log(res);
                    }).catch((err)=>{
                        console.log(err.message);
                    })
                    
                }).catch((err)=>{
                    setError(true);
                    setErr(err.response.data.message);
                })
            

        }
        
    }
    const checkUrl = ()=>{
    
        const obj = urlParser.parse(Url);
        console.log(obj);
        const providers = ['youtube','facebook','soundcloud','vimeo','wistia','mixcloud','dailymotion','twitch'];
        try{
            if(obj.mediaType === 'video'){
                setError(false);
                setErr('');
            }
            if(!obj.provider in providers){
                setErr(`${obj.provider} isn't supported`);
                return false;
            }
            setError(false);
            return true;
        }
        catch(err){
            setErr('Please enter a valid URL');
            return false;
        }
    }
    return(
        <div className = 'upload-bg'>
            <Header/>
            {Log?<Redirect to = '/login'/>:<span/>}
            {Dash?<Redirect to = '/in'/>:<span/>}
            <div className = 'upload'>
                <div className = 'container'>
                    <div className = 'mediaup mt-5'>
                        <h2>Upload your media file</h2>
                        <a href = '/in'><CloseIcon fontSize = 'large' className = 'cross'/></a>
                    </div>
                </div>
                <div className = 'up-area'>
                    <form className = 'form' noValidate autoComplete="off" onSubmit = {()=>{checkUrl()}}>
                        <TextField id="outlined-basic" className = 'col-12 col-sm-12 ' label="Enter URL" type = 'url' variant="outlined" onChange = {(e)=>{
                            setUrl(e.target.value);
                        }}/>
                        <TextField id="outlined-basic" className = 'mt-5 col-12 col-sm-12' label="Video name" variant="outlined" onChange = {(e)=>{setName(e.target.value);}}/>
                        <TextField id="outlined-basic" className = 'mt-5 col-12 col-sm-12' spellCheck='true' label="What do you want to search for?" variant="outlined" onChange = {(e)=>{setQuery(e.target.value.toLowerCase());}}/>
                        
                        <FormControlLabel 
                            className = 'mt-5 mb-5'
                            onChange = {()=>{setCheck(!Check)}}
                            control = {<Checkbox color = 'secondary'/>}
                            label = "By checking this box, I certify that use of any facial recognition functionality in this service is not by or for a police department in the United States, and I represent that I have all rights (and individuals’ consents, where applicable) to use and store the file/data, and agree that it will be handled per the Online Services Terms and the Privacy Statement."
                            labelPlacement = 'end'
                        />
                        {Error ? <p className = 'error'><b>{Err}</b></p>:<p></p>}
                        {Check?<Button variant = 'contained'  className = 'mt-3' color = 'primary' style = {{width : '30%'}} onClick = {handleClick}>Upload</Button>:<Button variant = 'contained'  className = 'mt-3' color = 'primary' disabled style = {{width : '30%'}} onClick = {handleClick}>Upload</Button>}
                    </form>
                    <div >
                        
                        {Open?<Alert severity="success" style = {{left:0}}>
                            <h5>Video Uploaded Successfully. Please wait, you will be redirected shortly.</h5>
                        </Alert>:<span/>}
                        {Error? <Alert severity = "error"><h5>{Err}</h5></Alert>:<span/>}
                        {Upload? <Redirect to ={RedirectUrl}/>:<span/>}
                    </div>
                </div>
            </div>
        </div>
    );

}

export default Upload;