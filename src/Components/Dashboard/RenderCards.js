import React,{useState} from 'react';
import './Dashboard.css';
import Card from '@material-ui/core/Card';
import CardActionArea from '@material-ui/core/CardActionArea';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Typography from '@material-ui/core/Typography';
import ReactPlayer from 'react-player';


function RenderCards({url , name , query , id}){
    // const [Vids , setVids] = useState(false);
    
    let user_queries = `q=${name}&id=${id}`;
    
    return(
        <div>
            {/* {Vids?<Redirect to = {`/player/${user_queries}`}/>:<span/>} */}
            <a href = {`/player/${user_queries}`} style = {{textDecoration :'none'}}>
                <Card>
                <CardActionArea>
                    <CardMedia>
                        <ReactPlayer url = {url} width = '100%' height = '100%' controls/>
                    </CardMedia>
                    <CardContent>
                        <Typography gutterBottom variant="h5" component="h2" style = {{textTransform:'uppercase'}}>
                            {name}
                        </Typography>
                        <h3>You searched for:</h3>
                        <Typography variant="body2" color="textSecondary" component="p">
                            {query}
                        </Typography>
                    </CardContent>
                </CardActionArea>
            </Card>
            </a>

        </div>
    );
}

export default RenderCards;