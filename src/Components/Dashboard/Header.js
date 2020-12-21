import React,{useState , useEffect} from 'react';
import './Dashboard.css';
import { makeStyles } from '@material-ui/core/styles';
import { deepPurple } from '@material-ui/core/colors';
import HelpOutlineIcon from '@material-ui/icons/HelpOutline';
import Avatar from '@material-ui/core/Avatar';
import {Button} from '@material-ui/core';
import Axios from 'axios';
import Cookies from 'js-cookie';
import {Redirect } from 'react-router-dom';
import urls from '../../config.js';


const useStyles = makeStyles((theme) => ({
    root: {
      display: 'flex',
      '& > *': {
        margin: theme.spacing(1),
      },
    },
    purple: {
        color: theme.palette.getContrastText(deepPurple[500]),
        backgroundColor: deepPurple[500],
      },
  }));

function Header(){
    const classes = useStyles();
    const [Sign , setSign] = useState(false);
    const [User , setUser] = useState('');
    const [Email , setEmail] = useState(''); 
    const [Out , setOut] = useState(false);
    useEffect(()=>{
      Axios.get(`${urls.node_url}dashboard`,{withCredentials : true}).then((res)=>{
          setUser(res.data.user.name);
          setEmail(res.data.user.email);

        }).catch((err)=>{
            console.log(err);
            setOut(true);
        })
    },[]);
    return(
        <div className = 'header-bg'>
            <div className = 'header'>
                <h1 className = 'logo' style = {{fontSize : '40px' , color:'white' , cursor :'pointer'}}><a href = '/in' style = {{textDecoration:'none',color : 'white'}}>SeeVid!</a></h1>
                <HelpOutlineIcon fontSize = 'large' style = {{color : 'white' , cursor:'pointer'}} className = 'mr-3'/>
                <Avatar className = {classes.purple} aria-label = 'avatar' style = {{cursor : 'pointer'}} onClick = {()=>{setSign(!Sign)}}>{User.name}</Avatar>
            </div>
            {Sign?
            <div className = 'sign-out container'>
              <div className = 'row mt-1'>
                  <div className = 'col-12 col-sm-2'>
                    <Avatar className = {classes.purple} style = {{cursor : 'pointer'}}>{User.name}</Avatar>
                  </div>
                  <div className = 'ml-3 col-12 col-sm-6'>
                    <h3><b>{User}</b></h3>
                    <h3>{Email}</h3>
                  </div>

              </div>
              <Button variant = 'contained'  className = 'mt-3 mb-2' color = 'primary' style = {{width : '100%' , display : 'flex' , justifyContent:'center'}} onClick = {()=>{Cookies.remove('token' , {path:'/'}); setOut(true);}}>Sign Out</Button>
              {Out? <Redirect to = '/login'/>:<p/>}
            </div>
            :<div/>
            }
        </div>
    );
}

export default Header;