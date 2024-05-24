#venv/Scripts/activate
#Get-ExecutionPolicy -List
#Set-ExecutionPolicy -ExecutionPolicy /// -Scope /////

from flask import Flask,render_template,request

from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
import os

app = Flask(__name__)

@app.route('/',methods=['GET'])
def nice():
    return render_template('imgupd.html')

@app.route('/',methods=['POST'])
def predict():
    model=VGG16()
    img=request.files['image-file']
    img_path="./images/"+img.filename
    img.save(img_path)
     # Debugging: Print the current working directory
    print("Current working directory:", os.getcwd())

    # Debugging: List contents of the current directory
    print("Contents of the current directory:", os.listdir())

    # Debugging: List contents of 'images' directory
    images_dir = os.path.join(os.getcwd(), 'images')
    print("Contents of 'images' directory:", os.listdir(images_dir))


    image=load_img(img_path,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image=preprocess_input(image)
    
    y=model.predict(image)
    label=decode_predictions(y)
    label=label[0][0]
    classification=label[1]

    return render_template('biriyani.html',prediction=classification)

if __name__=="__main__":
    app.run(port=5500)
