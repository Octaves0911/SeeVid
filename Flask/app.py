# Importing Libraries
import time
import collections
from flask import Flask ,jsonify, request, json
from flask_cors import CORS
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import pafy
import youtube_dl
import easyocr   
print("Libraries Imported")

app = Flask(__name__)
CORS(app)

# declaring global variables
embedding_dimension = 256
units = 512
vocab_size = 20001
attention_features_shape = 64
max_length = 52



# Load the tokenizer saved as json
with open('tokenizer_json.json') as json_file:
    data = json.load(json_file)

tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
print("tokenizer loaded")

# Load the saved inception model that will be used to extract the initial features
image_feature_extract_model = tf.keras.models.load_model('image_feature_extract_model', compile = False)
print("Image feature Model Loaded")

# Load the vocab for the easyocr, here we are loading only english characters
reader = easyocr.Reader(['en'])
print("easy ocr vocab loaded") 

# Defining the Model classes

# Define the attention model
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                            self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        base_config = super().get_config()
        return{**base_config, "output_dim" : output_dim, "activation": activation}

# Define the decoder model that will use the above defined attention model
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# Define the decoder that will take the input from the inception model and encode it
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# Creating the class instances
encoder = CNN_Encoder(embedding_dimension)
decoder = RNN_Decoder(embedding_dimension, units, vocab_size)

# Optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                                                            from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Loading the checkpoints from the specified directory
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                        decoder=decoder,
                        optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Searching fot the latest checkpoint and restoring it
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)
print("checkpoints loaded")


# function to preprocess the image
def load_image(image):
    img = tf.convert_to_tensor(image)
    img = tf.image.resize(img, (229,299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

# Combining all the defined model to get the expected prediction
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image), 0)
    img_tensor_val = image_feature_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

     #   attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<eos>':
            return result,attention_plot
        

        dec_input = tf.expand_dims([predicted_id], 0)

 #   attention_plot = attention_plot[:len(result), :]
    return result#, attention_plot


# render each frame of the video and feed that to the model and save the outputs to a dict
def start_rendering(video_url):
    url = video_url
    vpafy = pafy.new(url)
    play = vpafy.getbest( preftype='mp4')
    cap = cv2.VideoCapture( play.url)
    frame_to_caption = collections.defaultdict(list)
    while (True):
        # extract each frame of the video
        ret,frame = cap.read()
        if (not ret):
            break

        f = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        s = " "

        if(frame is not None ):
            begin = time.time()
            #result=s.join(evaluate(frame)) 
            # get the caption of the current frame and store it
            result = evaluate(frame)
            frame_to_caption[f].extend( result[0])

            print(result[0])
            inter = time.time()
            print("Captioning Part over in ", (inter-begin))
            # get the text in the current frame and store that to the same dictionary
            text_in_frame = reader.readtext(frame)

            if(len(text_in_frame) > 0 ):
                text_list = s.join([x[1] for x in (text_in_frame)])
                print(text_list)
                frame_to_caption[f].extend(map(str.lower,text_list.split()))
                print("OCR part over in ",(time.time()-inter))
                print(frame_to_caption[f])
            print(f)
        

    cap.release()
    cv2.destroyAllWindows()
    return frame_to_caption


# function to convert the dictionary of frames and text to timestamps
def frame_to_timestamps(frame_to_caption,keyword):
    res = []
    for key, value in frame_to_caption.items():
        if(keyword in value):
            res.append(int(key))

    res = set(res)
    res = list(res)
    res.sort()
    min = 0
    max = res[-1]
    final_stamps = []

    for i in range(len(res)-1):
        if (i==0):
            min = res[i]

        if((res[i+1]-res[i] >3)):
            max = res[i]
            final_stamps.append([min,max])
            min = res[i+1]
        max = res[i+1]
    final_stamps.append([min,max])
    return final_stamps


@app.route('/')
def helloWorld():
    return "Heroku server up and running"


# route to get the url and keyword from the frontend and execute the required functions 
frame_to_captions = collections.defaultdict(list)

@app.route('/videourl',methods = ['POST', 'GET'])
def variable():
    time_stamps = [[]]
    token  = ''
    if request.method== 'POST':
        print("video url and keyword")
        url = request.data
        url_json = url.decode("utf-8")
        video_url = json.loads(url_json)
        real_video_url = video_url['url']
        keyword = video_url['Query']
        
        if real_video_url != "":
            frame_to_captions = start_rendering(real_video_url)
            time_stamps= frame_to_timestamps(frame_to_captions,keyword)
            token = "True"
        return jsonify({"timestamp": time_stamps , "description": frame_to_captions})
    elif request.method== "GET":
        if token != '':
            return jsonify({"token": "True"})
        else:
            return jsonify({"token": "False"})
      

@app.route("/search")
def search():
    print("new keyword is inserted")
    time_stamps = [[]]
    frame=request.data
    frame_json =frame.decode("utf-8")
    frame_to_caption_dictionary = json.loads(frame_json)
    keyword = frame_to_caption_dictionary['topic']
    time_stamps= frame_to_timestamps(frame_to_captions,keyword)
    return jsonify({"timestamp": time_stamps})

if __name__ == "__main__":
    app.run(debug=False)