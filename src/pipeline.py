from preprocessing import load_images,preprocessing_images
from training import CNN_training,save_model
from evaluation import CNN_evaluate
from sklearn.model_selection import train_test_split


Brain_Tumor = "/home/shiva/Desktop/ML/Files/Brain Tumor Prediction/Data/Brain Tumor"
Healthy ="/home/shiva/Desktop/ML/Files/Brain Tumor Prediction/Data/Healthy"

path_list = [Brain_Tumor, Healthy]
class_labels = ['BrainTumor', 'Healthy']


def pipeline(path_list):
    
    data = load_images(path_list,class_labels)
    
    train_df,test_df = train_test_split(data, test_size=0.20, random_state=30,stratify=data.labels)
    
    train,test = preprocessing_images(train_df,test_df)
    
    model = CNN_training()
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(train,epochs=6,batch_size=32,validation_data=test,shuffle=True)
    
    save_model(model)
    
    loss,accuracy = CNN_evaluate(model,test)
    
    print("losss ",loss)
    print("accuracy",accuracy)
    
if __name__== '__main__':
    pipeline(path_list)
        

    
    
    
    
    
    
    
    
    
    