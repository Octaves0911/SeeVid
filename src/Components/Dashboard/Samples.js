import React from 'react';
import './Dashboard.css';
import ReactPlayer from 'react-player';
import RenderCards from './RenderCards.js';

function Samples (){
    let url = ['https://www.youtube.com/watch?v=jzD_yyEcp0M&list=RDCwfoyVa980U&index=5&ab_channel=Marshmello','https://www.youtube.com/watch?v=tk36ovCMsU8&list=RDtk36ovCMsU8&start_radio=1&ab_channel=Marshmello','https://www.youtube.com/watch?v=7wtfhZwyrcc&list=RDtk36ovCMsU8&index=3','https://www.youtube.com/watch?v=V1Pl8CzNzCw','https://www.youtube.com/watch?v=ysz5S6PUM-U','https://www.youtube.com/watch?v=2Vv-BfVoq4g'];
    const vids = url.map((vid)=>{
        return(
                <div className = 'col-12 col-sm-3 mb-4'>
                    <RenderCards url={vid} name='Sample' query = 'sampless'/>
                    {/* <ReactPlayer url={vid} width = '100%' height = '100%' muted/>         */}
                </div>
        );
    })
    return(
        <div className = 'library container mt-5'>
            <div className = 'row'>

                {vids}
            </div>
        </div>
    );
}

export default Samples;