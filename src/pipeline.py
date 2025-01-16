from preprocessing import load_images,preprocessing_images,load_from_database,load_eval_images,load_test_from_database
from training import CNN_training,save_model
from evaluation import CNN_evaluate
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
import numpy as np  



Brain_Tumor = "/home/shiva/Desktop/ML/Files/Data/Brain Tumor"
Healthy ="/home/shiva/Desktop/ML/Files/Data/Healthy"

glioma = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/glioma"
meningioma ="/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/meningioma"
notumor = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/notumor"
pituitary = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/pituitary"

glioma_t = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Testing/glioma"
meningioma_t ="/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Testing/meningioma"
notumor_t = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Testing/notumor"
pituitary_t = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Testing/pituitary"


path_list = [glioma,meningioma,notumor,pituitary]

class_labels = ['glioma ','meningioma','notumor','pituitary']

path_list_test = [glioma_t,meningioma_t,notumor_t,pituitary_t]
load_dotenv()

url = os.getenv('url')

def pipeline(path_list):
    
    # train_df = load_images(path_list,class_labels)
    
    train_df = load_from_database(url)
    
    # test_df = load_images(path_list_test,class_labels)
    
    test_df = load_test_from_database(url)
    
    
    # train_df,test_df = train_test_split(d, test_size=0.20, random_state=30,stratify=data.labels)
    
    train,test = preprocessing_images(train_df,test_df)
    

    model = CNN_training()
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(train,epochs=7,batch_size=32,validation_data=test,shuffle=True)
    
    save_model(model)
    
    loss,accuracy = CNN_evaluate(model,test)
    print("loss",loss)
    print("accuracy",accuracy)
    
if __name__== '__main__':
    pipeline(path_list)
        

    
    
    
    
    
    
    
    
    
    