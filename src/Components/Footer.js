import { grey } from '@material-ui/core/colors';
import React from 'react';
import PhoneIcon from '@material-ui/icons/Phone';
import MailIcon from '@material-ui/icons/Mail';
import FacebookIcon from '@material-ui/icons/Facebook';
import InstagramIcon from '@material-ui/icons/Instagram';
import TwitterIcon from '@material-ui/icons/Twitter';

function Footer(){
    return(
        <div className = 'footer padd-u'>
            <div className = 'footer-layer1 container'>
                <div className = 'row footer-row'>
                    <div className = 'col-12 col-sm-6 left'>
                        Great for Editors , Streamers  and Bussinessmen<br/><PhoneIcon className = 'social-footer'/><MailIcon className = 'social-footer ml-2'/>
                    </div>
                    <div className ='col-12 col-sm-6 options'>
                        <h3 className = 'mt-4' style = {{color: grey}}>Menu</h3>
                        <h4 className = 'mt-2'>Get Started</h4>
                        <h4>Overview</h4>
                        <h4>Tutorial</h4>
                        <h4>About</h4>
                    </div>
                </div>
                <div className = 'row mt-5 end'>
                    <div className = 'col-12 col-sm-6'>
                            <InstagramIcon fontSize = 'large' className = 'social-footer'/>
                            <FacebookIcon fontSize = 'large' className = 'ml-2 social-footer'/>
                            <TwitterIcon fontSize = 'large' className = 'ml-3 social-footer'/>
                    </div>
                    <div className = 'col-12 col-sm-6 copyright'>
                        Copyright © 2020 SeeVid. All rights reserved.
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Footer;