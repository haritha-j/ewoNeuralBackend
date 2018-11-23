from flask import Flask, request
import demo
import os
from demo import infer
UPLOAD_FOLDER = '/media/haritha/Stuff/code/automation 2/pose/keras_Realtime_Multi-Person_Pose_Estimation/sample_images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    try:
        print (infer('sample_images/input.jpg'))
    except IndexError:
        print ("pose not recognized")
    return 'inference complete'

@app.route('/getImage', methods=['POST'])
def upload():
    try:
        imagefile = request.files['file']
        print("image recieved")
        print (imagefile)
        imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg"))
        print (infer('sample_images/input.jpg'))
        return 'inference complete'
        
    except Exception as err:
        print ('pose not recognized')
    return 'received'
