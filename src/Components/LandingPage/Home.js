import React from 'react';
import './Landing.css';
import Desc from './Desc.js';
import Footer from '../Footer.js';

function Home (){
  
  return(
    <div>
      <div className = 'Home'>
        <Desc/>
      </div>
      <Footer/>
    </div>
  );
}

export default Home;