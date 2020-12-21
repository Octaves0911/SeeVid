<p align="center">
  <img src="https://cdn.discordapp.com/attachments/778287286591946765/790344814255669248/unknown.png" />
  </p>
  
  # SeeVid
  
Thinking of adding a magical sunset scene to end a reel? Maybe particular part of a conference that you missed? Whatever it maybe, SeeVid provides you the timestamp of the exact content you're looking for with just a prompt word. It serves by helping pinpoint these instances to you, make the task of extracting your favorite clip from a video hassle-free and thus conveniently aiding users in the creation of diverse content!


  # Problem Statement

Online videos would make up for over 80% of all consumer traffic. Considering how most people have been cooped up in their homes this year, it is safe to assume that this basis is accurate. More than 75% of people waste time watching other parts of the video, which they weren’t looking for. \
And there are tons of people who make these videos, for myriad reasons. It might be a student preparing a presentation, an influencer making a reel or a professional producing commissioned content. Let's not forget those who make these clips out of pure passion! There are new, up and coming apps everyday to help edit videos and pictures. There aren't, however, a lot of places to turn to when you want to escape foraging the boundless Internet for a few seconds of an hour long video. 


# What it does

* With this tool, the user will be able to maneuver to a particular clip by providing a link and description.
* Our user-friendly website first authenticates the user then takes in a request. 
* When the user enters a URL and a query, our AI model processes the request and returns the appropriate timestamp that satisfies the given query along with the duration of the clip.


# Built with
* TensoFlow 
  * Trained the deep learning model on TensorFlow using the coco2014 dataset to get the labels of the objects in each frame.
* PyTorch
  * the OCR module uses the Pytorch to make better detection/recognition as fast as possible.
* Flask  
  * used Flask to set up the server for the deep learning model
* React
  * front-end is built on React.js
* MongoDB
  * to store the user credentials, queries, processed video and other data
* Express
  * used express.js to create an api in order to connect the frontend and the database

# Future Scope

* Can be used as an tool for generating the insights of a video
* Execution time and accuracy can be improved
* more optimized similarity fucntions can be used for understanding the keywords better

# Contributors
* (Aman Kumar Mallik)[https://github.com/Octaves0911]
* (Rithik Dutt)[https://github.com/Rithik-rethink]
* (Ashutosh Mahapatra)[https://github.com/amahapatra13]


