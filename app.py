from flask import Flask,render_template,request,jsonify
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model



app = Flask(__name__)

model = load_model('cnn_model.keras')


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict",methods=['POST'])
def predict():
    
    image_url = request.form["url"]
    
    image = request.files['image-upload']
    
    if image_url!='':
        try:
            response = requests.get(image_url)
            img = np.array(bytearray(response.content),np.uint8)
        except:
            return render_template('home.html',prediction_text="Please upload valid image url.")    
            
    elif image !='':
        img_bytes = image.read()
        img = np.frombuffer(img_bytes, np.uint8)
    else:
        return render_template('home.html',prediction_text="file or link not uploaded.")    
             
    try: 
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(256,256))
        img= img.astype('float32')/255.0
        img = np.expand_dims(img,axis=0)
        pred = model.predict(img)
        print(pred)
        print(max(pred[0])*100)
        index = np.argmax(pred[0])
        print(index)
        labels = ['glioma','meningioma','not brain tumor','pituitary']
        
        if index==0:
           pred_text="The person is having brain tumor of type glioma."
           return jsonify({
              'labels':labels,
               'predictions': [float((max(pred[0])*100)),0,0,0],
                'predictionText':pred_text })
        elif index==1:
            pred_text= "The person is having brain tumor of type meningioma."
            return jsonify({
              'labels': labels,
               'predictions': [0,float((max(pred[0])*100)),0,0],
                'predictionText':pred_text })
        elif index==2:
             pred_text="The person is not having brain tumor"
             return jsonify({
              'labels': labels,
               'predictions': [0,0,float((max(pred[0])*100)),0],
                'predictionText':pred_text })
        else:
             pred_text="The person is having brain tumor of type pituitary"
             return jsonify({
              'labels': labels,
               'predictions': [0,0,0,float((max(pred[0])*100))],
                'predictionText':pred_text })


        
    except Exception  as e:
        print(e)
        return  render_template('home.html',prediction_text='Error in preprocessing the image')
    

    
    
if __name__ == '__main__':  
    app.run(debug=True)
