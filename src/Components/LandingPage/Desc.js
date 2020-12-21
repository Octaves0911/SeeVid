import React from 'react';
import './Landing.css';
import ArrowForwardIcon from '@material-ui/icons/ArrowForward';
import InstagramIcon from '@material-ui/icons/Instagram';
import FacebookIcon from '@material-ui/icons/Facebook';
import TwitterIcon from '@material-ui/icons/Twitter';
import vid from '../media/home.jpg';
import ml from '../media/ml.jpg';
import landing from '../media/landing.jpg';
import upload from '../media/uploading.svg';
import result from '../media/result.png';
import processing from '../media/processing.png';
import Zoom from 'react-reveal/Zoom';
import Fade from 'react-reveal/Fade';


function Desc(){
    return(
        <div className = 'desc'>
            <div className = 'container container-c'>
                <div className = 'row padd-u'>
                    <h1 className = 'col-12 logo' >SeeVid!</h1>
                </div>
                <div className = 'row socials'>
                <Zoom top opposite cascade>
                    <div className = 'col-12'>
                        <InstagramIcon className = 'handles ml-2' fontSize = 'large' style = {{opacity:1}}/>
                        <FacebookIcon className = 'handles ml-2' fontSize = 'large' style = {{opacity:1}}/>
                        <TwitterIcon className = 'handles ml-2' fontSize = 'large' style = {{opacity:1}}/>
                    </div>
                </Zoom>
                </div>
                <div className = 'row'>
                <Fade left opposite cascade>

                    <div className = 'col-12 col-sm-6 des'>

                        <h1 className = 'info'>SeeVid</h1>
                            <h2><b>

                                Perceive,
                            what you desire
                            </b>
                            </h2><br/>
                        <p className = 'explaination' >Thinking of adding a magical sunset scene to end a reel? Maybe particular part of a conference that you missed? Whatever it maybe, SeeVid provides you the timestamp of the exact content you're looking for with just a prompt word. It serves by helping pinpoint these instances to you, make the task of extracting your favorite clip from a video hassle-free and thus conveniently aiding users in the creation of diverse content!</p>
                        <h3 className = 'scroll-button mt-5'>Get Started <a href = '/login' className ='btn scroll-button-icon'><ArrowForwardIcon/></a></h3>
                    </div>
                </Fade>
                <Fade right>

                    <div className = 'col-12 col-sm-6'>
                        <img src = {vid} alt="" width = '100%' height = 'auto' style = {{borderRadius:'25px'}}/>
                    </div>
                </Fade>
                </div>
                <div className = 'row mt-5 padd-b' id = 'section-2' >
                    <Fade left>

                    <div className = 'col-12 col-sm-6'>
                        <img src = {ml} alt="" width = '100%' height = 'auto' style = {{borderRadius:'25px'}}/>
                    </div>
                    </Fade>
                    <Fade right opposite cascade>

                    <div className = 'col-12 col-sm-6 overview'>
                        <p className = 'headings'>overview</p>
                        <h1 className = 'info'>Why SeeVid?<br/>
                            </h1>
                        <p className = 'explaination' >Online videos would make up for over 80% of all consumer traffic. Considering how most people have been cooped up in their homes this year, it is safe to assume that this basis is accurate. More than 75% of people waste time watching other parts of the video, which they weren’t looking for.
And there are tons of people who make these videos, for myriad reasons. It might be a student preparing a presentation, an influencer making a reel or a professional producing commissioned content. Let's not forget those who make these clips out of pure passion! There are new, up and coming apps everyday to help edit videos and pictures. There aren't, however, a lot of places to turn to when you want to escape foraging the boundless Internet for a few seconds of an hour long video.</p>
                    </div>
                    </Fade>
                </div>
                <div className = 'row'>
                    <div className = 'col-12'>
                        <h1 className = 'goal'>Be it your online classes, your conference, or your content, SeeVid lets you find better.
                        </h1>
                    </div>
                </div>
                <div className = 'steps container'>
                    <div className = 'row'>
                        <div className = 'step-1 col-12 col-sm-4'>
                            <img src = {upload} width = '100%' height = '415vh' alt=""/>
                            <div className = 'step-desc'>
                                <h1 style = {{textAlign:'center' , fontWeight:600}} className = 'mt-4'>Upload the file</h1>
                                <h3 style = {{textAlign:'center'}}>Upload the youtube link of the video, its that simple!</h3>
                            </div>
                        </div>
                        <div className = 'step-2 col-12 col-sm-4'>
                            <img src = {processing} alt= '' width = '100%' height = '415vh' style={{paddingTop:'2px'}}/>
                            <div className = 'step-desc'>
                                <h1 style = {{textAlign:'center' , fontWeight:600}} className = 'mt-4'>Processing..</h1>
                                <h3 style = {{textAlign:'center'}}>Enter the keyword that you want to look for, and we will do the rest</h3>
                            </div>
                        </div>
                        <div className = 'step-3 col-12 col-sm-4'>
                            <img src = {result} alt= '' width = '100%' height = '415vh' />
                            <div className = 'step-desc'>
                                <h1 style = {{textAlign:'center' , fontWeight:600}} className = 'mt-4'>Watch the video</h1>
                                <h3 style = {{textAlign:'center'}}>Extract your favorite clip from a video hassle-free</h3>
                            </div>
                        </div>
                    </div>
                </div>
                  
            </div>
                {/* <div className = 'overlay-img'>
                    <img src = {landing} alt="" width = '100%' height = '100%' />
                    <h1 className = 'centered' >Lorem Ipsum is simply dummy text of the printing and typesetting industry.</h1>
                </div> */}
        </div>
    );
}

export default Desc;