import React,{useEffect, useState} from 'react';
import './Dashboard.css';
import Header from './Header.js';
import ReactPlayer from 'react-player';
import Axios from 'axios';
import {Redirect} from 'react-router-dom';
import Chip from '@material-ui/core/Chip';
import AccessTimeIcon from '@material-ui/icons/AccessTime';
import RenderCards from './RenderCards.js';
import Footer from '../Footer.js';
import Load from './Load.js';
import urls from '../../config.js';



function Player(){
    const [Log , setLog] = useState(false);
    const [Url , setUrl] = useState('');
    const [Query , setQuery] = useState('');
    const [name , setName] = useState('');
    const [Data , setData] = useState(false);
    const [lists , setLists] = useState([]);
    const [Error , setError] = useState(false);
    const [timestamp , setTimestamp] = useState([[]])
    const ref = React.createRef()

    useEffect(()=>{
        let link = window.location.pathname.split('=')[2];
        let params = {
            "link" : link
        }
        Axios.get(`${urls.node_url}dashboard`,{withCredentials : true}).then((res)=>{
            setLog(false);
        }).catch((err)=>{
            setLog(true);
        })
        Axios.post(`${urls.node_url}player`,params,{withCredentials:true}).then((res)=>{
            let url = res.data.results.url.split('&')[0];
            setUrl(url);
            setQuery(res.data.results.query);
            setName(res.data.results.name);
            setData(true);
            setTimestamp(res.data.results.timestamp)
            res.data.playlists.shift();
            setLists(res.data.playlists);
        }).catch((err)=>{
            console.log(err.message);
            setError(true);
        });

    },[])
    const times =  timestamp.map((time)=>{
        let div1 = Math.floor(time[0] / 60);
        let rem1 = time[0] % 60;
        let div2 = Math.floor(time[1] / 60);
        let rem2 = time[1] % 60;
        return(
            <div className = 'ml-3'>
                <Chip size="large"
                    icon={<AccessTimeIcon/>}
                    clickable
                    color = 'primary' label = {<h2>{div1}:{rem1} - {div2}:{rem2}</h2>} onClick={() => ref.current.seekTo(time[0])}></Chip>
            </div>
        );
    })
    const analysis = ()=>{
        return(
            <div>
                <h2>TimeStamps found for "{Query}":</h2><br/>
                <div className = 'row'>
                    {times}
                </div>
            </div>
        );
    }
    const playlist = lists.map((vid)=>{

        return(
            <div className = 'mb-3'>
                <RenderCards url = {vid.url.split('&')[0]} name = {vid.name} query = {vid.query} id = {vid._id}></RenderCards>
            </div>
        );
    })
    return(
        <div className = 'player-bg' style = {{height : 'auto'}}>
            {Log?<Redirect to = '/login'/>:<span/>}
            {Error? <Redirect to ='/error'></Redirect>:<span/>}
            <Header/>
            <div>
                <div style = {{height : 'auto'}} className = 'container mt-5'>
                    <div className = 'row'>
                        <div className = 'col-12 col-sm-9'>
                            {Data?<ReactPlayer url = {Url} controls width = '100%' height = '100vh' ref = {ref}></ReactPlayer>:<Load/>}
                            <h1 style = {{fontSize : '65px' , fontWeight:'500'}}><b>{name}</b></h1><br/>
                            <h2><b>Search Results for: {Query}</b></h2>
                            {Data? analysis() : <Load/>}
                        </div>
                        <div className = 'col-12 col-sm-3' style = {{overflow : 'scroll'}}>
                            <h1 className = 'player_name' style = {{fontSize : '30px' , fontWeight : '900',textAlign:'left',letterSpacing:'1px',textTransform:'uppercase'}}>Your Videos</h1>
                            {playlist}
                        </div>
                    </div>
                </div>
                {/* <div className = 'container details mt-5'>
                    <div className = 'row'>

                        <div className = 'col-12 col-sm-8'>
                        </div>
                    </div>
                </div> */}
            </div>
            <Footer/>
        </div>
    );
}

export default Player;