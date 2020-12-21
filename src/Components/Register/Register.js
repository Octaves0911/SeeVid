import React from 'react';
import './Register.css';
import {TextField} from '@material-ui/core';
import {Button} from '@material-ui/core';
import {FormControlLabel,Checkbox } from '@material-ui/core';
import ErrorOutlineIcon from '@material-ui/icons/ErrorOutline';
import Axios from 'axios';
import {Redirect} from 'react-router-dom';
import urls from '../../config.js';

class Register extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            username : "",
            email : "",
            confirm_email : "",
            password : "",
            error : null,
            code : 1,
            logged_in : false
        }
    }
    componentDidMount(){
        Axios.get(`${urls.node_url}dashboard`,{withCredentials : true}).then((res)=>{
            this.setState({
                logged_in : true
            })
        }).catch((err)=>{
            console.log(err.message);
        })
    }

    handleChange(event , element){
        var value = event.currentTarget.value;
        if(element === 'email'){
            this.setState({
                email : value
            })
        }
        else if(element === 'confirm_email'){
            this.setState({
                confirm_email : value
            })
        }
        else if(element === 'password'){
            this.setState({
                password : value
            })
        }
        else if(element === 'username'){
            this.setState({
                username : value
            })
        }
        
    }
    handleClick(event){
        var params = {
            "username" : this.state.username,
            "email" : this.state.email,
            "password" : this.state.password
        }
        const url =`${urls.node_url}register` 
        Axios.post(url , params, {
            "headers" : {
                "Accept" : "application/json",
                "content-type" : "application/json",
            },
            withCredentials : true
        }).then(response => {
            this.setState({
                code : response.data.code
            })
        }).catch(err => {
            console.log(err);
            this.setState({
                error : err.response.data.message
            })
        });
    }
    render(){
        if(this.state.logged_in){
            return(<Redirect to = "/in"/>);
        }
        if(this.state.code === 0){
            return(<Redirect to={{pathname: "/in", state: {token: this.state.token}}}/>);
        }
        return(
            <div className = 'container register'>
                <h1 className = 'logo2'>SeeVid</h1>
                <div className = 'row justify-content-center'>
                    <h1 className = 'col-12 col-sm-6'>Sign up for free to start Exploring.</h1>
                </div>            
                <div className = 'row justify-content-center'>
                    <form className = 'form' noValidate autoComplete="off">
                        <TextField id="outlined-basic" className = 'col-12 col-sm-12 ' label="What's your email?" variant="filled" onChange = {(event)=>this.handleChange(event,"email")}/>
                        <TextField id="outlined-basic" className = 'mt-5 col-12 col-sm-12' label="Confirm your email" variant="filled" onChange = {(event)=>this.handleChange(event,"confirm_email")} />
                        {this.state.email !== this.state.confirm_email ? <p className = 'col-12 col-sm-12 error2'><b><ErrorOutlineIcon color="secondary"/> The email addresses don't match.</b></p>:<p></p>}
                        <TextField id="outlined-basic" className = 'mt-5 col-12 col-sm-12' label="Create a password" variant="filled" type="password" onChange = {(event)=>this.handleChange(event,"password")}/>
                        <TextField id="outlined-basic" className = 'mt-5 col-12 col-sm-12' label="What should we call you?" variant="filled" onChange = {(event)=>this.handleChange(event,"username")}/>
                        <p className = 'mt-2 col-12 col-sm-12'><b>This appears on your profile*</b></p>
                        
                        <FormControlLabel 
                            control = {<Checkbox color = 'secondary' />}
                            label = "By clicking here, you agree to SeeVid's Terms and Conditions of Use"
                            labelPlacement = 'end'
                        />
                        <p className ='policy'>To learn more about how Spotify collects, uses, shares and protects your personal data please read SeeVid's <a href = '/'>Privacy Policy.</a></p>
                        {this.state.error !== null ? <p className = 'error'><b><ErrorOutlineIcon color="secondary"/> {this.state.error}</b></p>:<p></p>}
                        <Button variant = 'contained'  className = 'mt-3' color = 'primary' style = {{width : '100%'}} onClick = {this.handleClick.bind(this)}>Sign In</Button>
                        <p className = 'login_redirect mt-2'>Have an account? <b><a href = '/login'>Log in.</a></b></p>
                    </form>

                </div>
            </div>
        );
    }
}

export default Register;