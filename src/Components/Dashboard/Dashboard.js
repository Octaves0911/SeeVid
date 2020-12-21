import React,{useState , useEffect} from 'react';
import './Dashboard.css';
import Header from './Header.js';
import Footer from '../Footer.js';
import Samples from './Samples.js';
import Library from './Library.js';
import PublishIcon from '@material-ui/icons/Publish';
import Axios from 'axios';
import {Redirect} from 'react-router-dom';
import urls from '../../config.js';


  function Dashboard(){
      const [Lib , setLib] = useState(true);
      const [Sam , setSam] = useState(false);
      const [Log , setLog] = useState(false);

      useEffect(()=>{
          Axios.get(`${urls.node_url}dashboard`,{withCredentials : true}).then((res)=>{
              setLog(false);
            }).catch((err)=>{
                setLog(true);
            })
        },[]);
        
        return(
            <div className = 'dashboard'>
            <Header/>
            {Log?<Redirect to = '/login'/>:<div/>}
                <div className ='container'>
                    <div className = 'jumbo'>
                        <h1>Extract insights and enhance your content</h1>
                        <h3>Search for any text, person, insight, or object in your videos</h3>
                        <a href = '/up' className ='btn mt-5'><PublishIcon/>Upload</a>
                    </div>
                    <div className = 'row mt-5'>
                        <div className = 'col-12 col-sm-12 hero'>
                            {Lib?<h3 className = 'active options'>Library</h3>:<h3 className = 'options' onClick = {()=>{setLib(true) ; setSam(false)}}>Library</h3>}
                            {Sam?<h3 className = 'active options'>Discover</h3>:<h3 className = 'options' onClick = {()=>{setSam(true); setLib(false);}}>Discover</h3>}

                        </div>
                    </div>
                </div>

                <div className = 'sub'>
                    {Lib?<Library/>:<Samples/>}
                </div>
            <Footer/>
        </div>
    );
}

export default Dashboard;