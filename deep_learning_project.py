import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import cm
import librosa
import csv
import os
import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


#The following code for feature extraction from audiofiles and data preprocessing was adopted and adapted from
#https://github.com/ravasconcelos/spoken-digits-recognition/blob/master/src/spoken-digits-recognition.ipynb
#Function to extract spectral features from audio file and save it in a csv file
def feature_extraction(folder_with_audio, file_name_csv):
    print("Starting feature extraction for {0} data files".format(folder_with_audio))
    print("The features of the files in the folder \"{0}\" will be saved to {1}".format(folder_with_audio,file_name_csv))
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 129):
        header += f' mfcc_mean{i} mfcc_median{i} mfcc_std{i}'
    header = header.split()
    file = open(file_name_csv, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(header)
    files_list=os.listdir(folder_with_audio)
    for current,filename in enumerate(files_list):
        print(f'Processing{current} out of {len(files_list)}...')
        number = f'{folder_with_audio}/{filename}'
        y, sr = librosa.load(number, mono=True, duration=30)
        # remove leading and trailing silence
        y, index = librosa.effects.trim(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=129)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)} {np.median(e)} {np.std(e)}'
        writer.writerow(to_append.split())
    file.close()
    print("End of feature extraction for {0} data".format(folder_with_audio))

#Function to preprocess the training data
def preProcessData(file_name_csv):
    print(file_name_csv+ " will be preprocessed")
    data = pd.read_csv(file_name_csv)
    filenameArray = data['filename'] 
    speakerArray = []
    for i in range(len(filenameArray)):
        #print(filenameArray[i])
        speaker = label[filenameArray[i]]
        speakerArray.append(speaker)
    data['number']= speakerArray
    #Dropping unnecessary columns
    data = data.drop(['filename'],axis=1)
    print("Preprocessing is finished")
    return data

#Function to preprocess test data
def preProcessTestData(file_name_csv):
    print(file_name_csv+ " will be preprocessed")
    Final={}
    data = pd.read_csv(file_name_csv)
    for i in data['filename']:
        Final[i]=''
    data = data.drop(['filename'],axis=1)
    return data,Final

#The following code was partly adopted and adapted from 
# https://github.com/ThejakaSEP/FeedForwardNet/blob/main/NeuralNet.py
#Neural network class to train and predict
class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.dense_layers = nn.Sequential(
            nn.Linear(390, 184),
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        logits = self.dense_layers(input_data)
        predictions = self.softmax(logits)
        return predictions

#The following code was partly adopted and adapted from 
# https://github.com/shansach/ssn_spinal/blob/main/dcase_cnn.py
#Function to train the model
def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        #Function to calculate loss and back propogate
        for input, target in data_loader:
            # calculate loss
            input, target = input.to(device), target.to(device)
            prediction = model(input.float())
            loss = loss_fn(prediction, target.long())
            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(f"loss: {loss.item()}")
        print("---------------------------")
    print("Finished training")   

#Function to prediction the output classes for test data    
def predict(model, inp):
    model.eval()
    with torch.no_grad():
        predictions = model(inp)
        predicted_index = predictions[0].argmax(0)
        predicted = predicted_index
    return predicted


path_to_file='train.json'
#Loading labels of training data
with open(path_to_file) as data_file:
    label = json.load(data_file)
labels=list(label.values())

#Check if the directory already contains the feature file if not, commence
# the feature extraction from audio files
# PS: feature extraction can take around 2hrs
if 'train.csv' in os.listdir():
    print("train.csv is available in current folder..skipped feature extraction")
else:
    feature_extraction("train", 'train.csv')
if 'test.csv' in os.listdir():
    print("test.csv is available in current folder..skipped feature extraction")
else:    
    feature_extraction("test", 'test.csv')

print("Preprocessing data started")
trainData = preProcessData('train.csv')
testData,Final = preProcessTestData('test.csv')
print("Preprocess finished")


# Hyper parameters for neural network model
BATCH_SIZE = 125
EPOCHS = 25
LEARNING_RATE = 0.001

X = np.array(trainData.iloc[:, :-1], dtype = float)
Y = np.array(trainData.iloc[:, -1], dtype = float)
Z=np.array(testData.iloc[:, :], dtype = float)

train_X,val_X,train_Y,val_Y=train_test_split(X,Y, test_size=0.2,stratify=Y)

#Standardizing the input features
scaler = StandardScaler()
X = scaler.fit_transform(train_X)
val_X=scaler.transform(val_X)
Z=scaler.transform( Z )

#Converting the dataset into tensors
inputs = torch.Tensor(X)
targets = torch.Tensor(train_Y)
val_input=torch.Tensor(val_X)
test=torch.Tensor(Z)

#Using dataloader to load the input data to NN
dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, BATCH_SIZE, shuffle = True)

#The following code was partly adopted and adapted from 
# https://github.com/shansach/ssn_spinal/blob/main/dcase_cnn.py

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    #Setting up Neural Network model    
    feed_forward_net = FeedForwardNet().to(device)
    
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    print("Model training has been started...")
    train(feed_forward_net, train_loader, loss_fn, optimiser, device, EPOCHS)
    print("Training completed")
    val_predicted=[]
    true=[]
    print("Starting validation")
    for z in range(len(val_input)):
        i=val_input[z]
        inp=i.unsqueeze(0)
        pred= predict(feed_forward_net, inp)
        pred=int(np.array(pred).ravel())
        exp=int(val_Y[z])
        val_predicted.append(pred)
        true.append(exp)
    print("Calculating error rate")
    ER=np.array([val_predicted[key] != true[key] for key in range(len(val_input))]).mean()
    print("Error rate",ER)
    print("Starting prediction for test files")
    predicted=[]
    for z in range(len(test)):
        i=test[z]
        inp=i.unsqueeze(0)
        pred= predict(feed_forward_net, inp)
        pred=int(np.array(pred).ravel())
        predicted.append(pred)
    for i,j in enumerate(Final.keys()):
        Final[j]=str(predicted[i])
    with open('test.json', 'w') as file:
        file.write(json.dumps(Final, indent=4))
    print("Prediction completed. Check directory for JSON file")




















'''
# The following seed technique has been commented out as it worsens the score
# hence it's impossible to reproduce the exact result every time so there could
# be a slight variance in final JSON file and score for every execution

torch.manual_seed(101)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
#The following code for seeds setting was taken from 
#https://madewithml.com/courses/foundations/utilities/
SEED = 101
def set_seeds(seed=101):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seeds(seed=SEED)'''
  