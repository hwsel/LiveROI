# -*- coding:utf-8 -*-

'''============================================================
Modification for G:

Fix user feedback error

============================================================='''
"""script for predicting labels from live feed"""
import sys
import heapq
sys.path.append('/python')
import numpy as np
import caffe
import cv2
import math
import scipy.io as sio
import time as tm
import random
import itertools
import csv
from line_profiler import LineProfiler

TestWords = False




# =========import for phrase2vec============

if TestWords == True:
    import sys
    # reload(sys)
    # sys.setdefaultencoding("utf-8")

    ######################
    # Loading word2vec
    ######################

    # import phrase2vec
    import gensim
    from gensim.models import word2vec

    # Change this to your own path.
    pathToBinVectors = 'GoogleNews-vectors-negative300.bin'

    print "Loading the data file... Please wait..."
    model1 = gensim.models.KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
    # model1 = gensim.models.Word2Vec.load(pathToBinVectors)
    print "Successfully loaded 3.6 G bin file!"

# How to call one word vector?
# model1['resume'] -> This will return NumPy vector of the word "resume".

import numpy as np
import math
from scipy.spatial import distance

from random import sample
import sys
from nltk.corpus import stopwords

# =========import for phrase2vec  end ============

batch_size = 16  # number of samples per video
# !/usr/bin/env python

import thread
import threading
import time
import os
import threading, time, signal
import sys

PublicData = 4
Switch = False
N_tile = 5

'''
global varibles defination

'''
flag_Count = 0
running_frames_T = []
last_16_frames_T = []
Predict_result = []
Predict_result_Score = []
Predict_result_Top5 = []
Predict_result_Top5_Score = []
Caffe_net = []
Flag_working = []

UserModelListW=[]   
UserModelList=[]    
UserModel_5=[]      
ConstModelLen=30   



def userLoca_One(FrameRate, FileNameStart, i):
    """
    :param FrameRate: FPS
    :param FileNameStart: The filename will be generated with FileNameStart and the "i"
    :param i: the number of the user
    :return: Userdata
    """

    '''

    cap = cv2.VideoCapture(video_name)
    FrameRate = int(round(cap.get(5)))      # 29.9 fps changed to 30
    TotalFrames = cap.get(7)
    print ("frame rate is: ",FrameRate,"  Total frames is: ",TotalFrames)
    '''

    '''
        Read user data file and collect all records
        Save them in two lists:
        Userdata and TimeStamp
        Userdata is where store the user location (convert to fload)
        TimeStamp is for syncronization

    '''
    VideoName = "video_0"
    DirectorName = "\"
    i = 1
    
    UserFile = DirectorName + VideoName + '_' + str(i) + ".csv"
    Userdata = []
    flagTime = 1
    TimeStamp = []
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)  
        birth_header = next(csv_reader)  
        for row in csv_reader:  
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])
    flagTime = 0
    Userdata = [[float(x) for x in row] for row in Userdata] 
    Userdata = np.array(Userdata)  
    TOT_Len = len(Userdata)
    print(TOT_Len)
    print(Userdata.shape)
    NUmberCount = 0
    #just in one second
    while NUmberCount < TOT_Len:
        #load your data to UserAll
        UserLocationPerFrame.append(UDATA)
    return UserLocationPerFrame

def GetActionFromeResult_SingleB(Predict_result_Top5):
    # userInput1 = raw_input("Type the phrase1: ")
    # print (userInput1)
    # AA = "woqu"
    # print (AA)
    EE = Predict_result_Top5
    IN_K = EE.index('\'')
    CC = EE[:-1]
    # print ("IN_K====>", IN_K)
    BB = CC[IN_K + 1:]
    # print (BB)
    return BB

def GetActionFromeResult_Single(Predict_result_Top5):
    # userInput1 = raw_input("Type the phrase1: ")
    # print (userInput1)
    # AA = "woqu"
    # print (AA)
    EE = Predict_result_Top5[0][2]
    IN_K = EE.index('\'')
    CC = EE[:-1]
    # print ("IN_K====>", IN_K)
    BB = CC[IN_K + 1:]
    # print (BB)
    return BB


def UpdateOnUserFeedback():
    return True



def UpdateUserModel(UserList,ObjectList):
    UserList[0]=1




def Demoheatmapillustration():
    ProceList = []
    VideoUserName = "video_3_"
    TimeStamp = []
    flagTime = 1

    
    A = 1  # 47                    #49
    for i in range(1, A + 1):  # 49
        UserFile = VideoUserName + str(i) + ".csv"
        Userdata = []
        with open(UserFile) as csvfile:
            csv_reader = csv.reader(csvfile)  
            birth_header = next(csv_reader)  
            for row in csv_reader:  
                Userdata.append(row[1:])
                if flagTime == 1:
                    TimeStamp.append(row[0])
        flagTime = 0
        Userdata = [[float(x) for x in row] for row in Userdata]  
        Userdata = np.array(Userdata)  
        ProceList.append(Userdata)

    UserOverall = []
    strA = TimeStamp[0].split(':')
    PreTime = math.ceil(float(strA[2]))
    CurTime = math.floor(float(strA[2]))
    i = 0
    j = 0
    FrameWidthT = 1280
    FrameHeightT = 720
    Frame30 = 30
    FrameCount = Frame30 + 1

    # =============================
    if FrameCount >= Frame30:
        UserIn1s = []
        UserIn1sR = []
        Ind = 0
        # load your user data to UserAll
        while PreTime > CurTime:
            UserAll.append(UDATA)

        
        
        label = []
        for i in range(len(UserIn1sR)):
            label.append(check_box(UserIn1sR[i][0][0], UserIn1sR[i][0][1]))

        # label[check_box(UserIn1sR[0][0][0], UserIn1sR[0][0][1])]=1
    else:
        print'=='



class PhraseVector:
    def __init__(self, phrase):
        self.vector = self.PhraseToVec(phrase)

    # <summary> Calculates similarity between two sets of vectors based on the averages of the sets.</summary>
    # <param>name = "vectorSet" description = "An array of arrays that needs to be condensed into a single array (vector). In this class, used to convert word vecs to phrases."</param>
    # <param>name = "ignore" description = "The vectors within the set that need to be ignored. If this is an empty list, nothing is ignored. In this class, this would be stop words."</param>
    # <returns> The condensed single vector that has the same dimensionality as the other vectors within the vecotSet.</returns>
    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore=[]):
        if len(ignore) == 0:
            return np.mean(vectorSet, axis=0)
        else:
            return np.dot(np.transpose(vectorSet), ignore) / sum(ignore)

    def PhraseToVec(self, phrase):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector = model1[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)

    # <summary> Calculates Cosine similarity between two phrase vectors.</summary>
    # <param> name = "otherPhraseVec" description = "The other vector relative to which similarity is to be calculated."</param>
    def CosineSimilarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / (
                    np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity





def Prediction_Core_Thread(dims, running_frames, algo, image_mean, net, num_categories, index_to_label, Index_l):
    global flag_Count
    global Predict_result
    global Predict_result_Score
    global Predict_result_Top5
    global Predict_result_Top5_Score
    global N_tile
    initial_predictions = np.zeros((num_categories, 1))
    rgb = np.zeros(shape=dims, dtype=np.float64)
    # running_frames.append(last_16_frames)
    n_slots = len(running_frames)
    # print(running_frames)
    if (n_slots > 5):  # original is 5
        del running_frames[0]
        frames_algo = algo[4]
    else:
        frames_algo = algo[n_slots - 1]
    for y in range(len(frames_algo)):
        idx_frames = np.rint(np.linspace(0, len(running_frames[y]) - 1, frames_algo[y])).astype(np.int16)
        # print(idx_frames)
        running_frames[y] = [running_frames[y][i] for i in idx_frames]

    # last_16_frames = []
    flattened_list = list(itertools.chain(*running_frames))
    for ix, img_arr in enumerate(flattened_list):
        rgb[:, :, :, ix] = img_arr

    rgb_3 = rgb[16:240, 60:284, :, :]
    rgb = rgb_3
    rgb = rgb[...] - np.tile(image_mean[..., np.newaxis], (1, 1, 1, rgb.shape[3]))
    rgb = np.transpose(rgb, (1, 0, 2, 3))

    prediction = np.zeros((num_categories, 1))

    net.blobs['data'].data[...] = np.transpose(rgb[:, :, :, :], (3, 2, 1, 0))
    output = net.forward()
    prediction[:, :] = np.transpose(output["fc8"])
    predictions_mean = np.mean(prediction + initial_predictions, axis=1)

    initial_predictions = predictions_mean
    predict_ind = np.argmax(predictions_mean)
    predict_top5_ind = np.argpartition(predictions_mean, -5)[-5:]

    top5_label = [index_to_label[int(x)] for x in predict_top5_ind]
    
    del Predict_result_Top5[Index_l][0]
    del Predict_result_Top5_Score[Index_l][0]
    Predict_result_Top5[Index_l].append(top5_label)
    Predict_result_Top5_Score[Index_l].append(predict_top5_ind)
    # return predict_ind, top5_label
    # flag_Count+=1
    # print("len of Flag_working is :::" , len(Flag_working)," Index is ",i * N_tile + j)
    Flag_working[Index_l] = 0


def Prediction_Core(dims, running_frames, algo, image_mean, net, num_categories, index_to_label):
    initial_predictions = np.zeros((num_categories, 1))
    rgb = np.zeros(shape=dims, dtype=np.float64)
    # running_frames.append(last_16_frames)
    n_slots = len(running_frames)
    # print(running_frames)
    if (n_slots > 5):
        del running_frames[0]
        frames_algo = algo[4]
    else:
        frames_algo = algo[n_slots - 1]
    for y in range(len(frames_algo)):
        idx_frames = np.rint(np.linspace(0, len(running_frames[y]) - 1, frames_algo[y])).astype(np.int16)
        # print(idx_frames)
        running_frames[y] = [running_frames[y][i] for i in idx_frames]

    # last_16_frames = []
    flattened_list = list(itertools.chain(*running_frames))
    for ix, img_arr in enumerate(flattened_list):
        rgb[:, :, :, ix] = img_arr

    rgb_3 = rgb[16:240, 60:284, :, :]
    rgb = rgb_3
    rgb = rgb[...] - np.tile(image_mean[..., np.newaxis], (1, 1, 1, rgb.shape[3]))
    rgb = np.transpose(rgb, (1, 0, 2, 3))

    prediction = np.zeros((num_categories, 1))

    net.blobs['data'].data[...] = np.transpose(rgb[:, :, :, :], (3, 2, 1, 0))
    output = net.forward()
    prediction[:, :] = np.transpose(output["fc8"])
    predictions_mean = np.mean(prediction + initial_predictions, axis=1)

    initial_predictions = predictions_mean
    predict_ind = np.argmax(predictions_mean)
    predict_top5_ind = np.argpartition(predictions_mean, -5)[-5:]

    top5_label = [index_to_label[int(x)] for x in predict_top5_ind]
    return predict_ind, top5_label


def online_predict(mean_file, model_def_file, model_file, classes_file, num_categories):
    Buffer_Count = 0
    # ===========================================
    KeyWords = "cooking"
    if TestWords == True:
        phraseVector1 = PhraseVector(KeyWords)
    # ============parameters====================
    N_tile = 5
    # ===============for user model=============
    global UserModelList
    global UserModelListW
    global UserModel_5
    global ConstModelLen
    for i in xrange (4):
        UserModel_5.append(" ")
    # ==============list for each thread==========
    global flag_Count
    '''
    modify this to global for thread
    running_frames_T = []
    last_16_frames_T = []
    Predict_result = []
    Predict_result_Score=[]
    Predict_result_Top5 = []
    Predict_result_Top5_Score = []

    '''
    global running_frames_T
    global last_16_frames_T
    global Predict_result
    global Predict_result_Score
    global Predict_result_Top5
    global Predict_result_Top5_Score
    global Caffe_net
    global Flag_working
    Index_l = []
    TileStatus = []
    UserFeedBackLocation=[]
    for i in range(N_tile):
        for j in range(N_tile):
            Frame_Sub_List_Running = []
            Frame_Sub_List_16 = []
            running_frames_T.append(Frame_Sub_List_Running)
            last_16_frames_T.append(Frame_Sub_List_16)
            Predict_result.append('0')
            Predict_result_Score.append(0)
            Predict_result_Top5_sub = []
            Predict_result_Top5_sub.append(0)
            Predict_result_Top5.append(Predict_result_Top5_sub)
            Predict_result_Top5_Score_sub = []
            Predict_result_Top5_Score_sub.append(0)
            Predict_result_Top5_Score.append(Predict_result_Top5_Score_sub)
            Flag_working.append(1)
            Index_l.append(i * N_tile + j)
            TileStatus.append(0)
  
    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    frame_counter = 0
    index_to_label = {}

    # sampling scheme
    algo = [[16], [8, 8], [4, 4, 8], [2, 2, 4, 8], [1, 1, 2, 4, 8]]
    # algo = [[10],[5,5],[3,3,4],[2,2,3,3],[1,1,2,2,4]]
    with open(classes_file, "r") as file:
        for line in file:
            index, label = line.strip().split(" ", 1)
            index_to_label[int(index)] = label
    for i in range(N_tile):
        # for j in range(N_tile):
        net = caffe.Net(model_file, model_def_file, caffe.TEST)
        Caffe_net.append(net)
    '''
    video list and user location list and the matching between each other.
    '''
    VideoLen = 5400  # 5400
    # "1-7-Cooking BattleB","1-2-FrontB","1-6-FallujaB","1-8-FootballB","1-9-RhinosB",
    # "1-7-Cooking BattleB","1-2-FrontB","1-6-FallujaB","1-8-FootballB","1-9-RhinosB",
    # "2-1-KoreanB", "2-3-RioVRB",
    Video_FileList = ["1-1-Conan Gore FlyB","1-2-FrontB","1-6-FallujaB","1-7-Cooking BattleB","1-8-FootballB","1-9-RhinosB","2-1-KoreanB", "2-3-RioVRB","2-4-FemaleBasketballB", "2-5-FightingB", "2-6-AnittaB"]
    Video_FileListB= []
    i=0
    VideoFileBasic='//Video/'
   
    
    
    # video_name = '/home/kora/Downloads/1-7-Cooking Battle.mp4'
    cap = cv2.VideoCapture(video_name)
    capB = cv2.VideoCapture(video_name)
    W_Frame = cap.get(3)
    H_Frame = cap.get(4)
    # check the parameters of get frome:  
    print("===============frame width============")
    print (W_Frame)
    print("===============frame height============")
    print (H_Frame)
    FrameRate = int(round(cap.get(5)))  # 29.9 fps changed to 30
    TotalFrames = cap.get(7)
    TotalSeconds=int(round(TotalFrames/FrameRate))
    print("framerate and totalframes is:  ", FrameRate,TotalFrames)
    print("total second is: ",TotalSeconds)
    
    #FileNameStart="video_6_D1"
    LocationPerFrame=userLocal_One(FrameRate, FileNameStart, 1,TotalSeconds,H_Frame,W_Frame)
    print(len(LocationPerFrame[1]))
    #exit()
    # dims = (256,340,3,batch_size)
    dims = (256, 340, 3, batch_size)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)

    # show frame_num(time) and predictions
    text = ""
    time = ""

    d = sio.loadmat(mean_file)
    image_mean = d['image_mean']

    running_frames = []
    last_16_frames = []
    initial_predictions = np.zeros((num_categories, 1))

    # for 16 frames group
    
    last_16_frames_List = []
    for i in range(N_tile):
        for j in range(N_tile):
            last_16_frames_sub = []
            last_16_frames_List.append(last_16_frames_sub)
    FrameCount=0
    FrameCountDelay=0
    
    point_size = 7
    point_color = (0, 0, 255)  # BGR
    thickness = -1  #  0 、4、8
    PredictFlag=0    
    while (True):
        # Capture frame-by-frame
        time = "Frame: " + str(frame_counter)
        t0 = tm.time()
        
        ret, frame = cap.read()
        FrameCount+=1
        ret, frame = cap.read()
        FrameCount += 1
        ret, frame = cap.read()
        FrameCount += 1
        ret, frame = cap.read()
        FrameCount += 1
        Whole_Frame = frame.copy()
        # frame = frame[3*256:4*256+0,5*340+150:6*340+150]
        
        Frame_List = []
        for i in xrange(N_tile):
            for j in xrange(N_tile):
                if i == 0:
                    Height_L = i * 144
                    Height_H = (i + 1) * 144 + 112
                elif i == 4:
                    Height_L = i * 144 - 112
                    Height_H = (i + 1) * 144
                else:
                    Height_L = i * 144 - 56
                    Height_H = (i + 1) * 144 + 56

                if j == 0:
                    Weight_L = j * 256
                    Weight_H = (j + 1) * 256 + 84
                elif j == 4:
                    Weight_L = j * 256 - 84
                    Weight_H = (j + 1) * 256
                else:
                    Weight_L = j * 256 - 42
                    Weight_H = (j + 1) * 256 + 42

                Frame_Sub = frame[Height_L:Height_H, Weight_L:Weight_H]
                Frame_List.append(Frame_Sub)
                cv2.putText(Whole_Frame, Predict_result[i * N_tile + j], (j * 256 + 10, i * 144 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=1)
                
                if TileStatus[i * N_tile + j]==0:
                    ColorPrediction=(0, 255, 0)
                else:
                    ColorPrediction = (255, 0, 0)
                cv2.rectangle(Whole_Frame, ( j * 256,i * 144), ( (j+1) * 256,(i+1)* 144), ColorPrediction, 2)
                #FrameCount+=1
                if TestWords == True:
                    phraseVector2 = PhraseVector(Predict_result[i * N_tile + j])
                    similarityScore = phraseVector1.CosineSimilarity(phraseVector2.vector) * 10
                    print(KeyWords, Predict_result[i * N_tile + j], similarityScore)
                    cv2.putText(Whole_Frame, str(similarityScore), (j * 256 + 10, i * 144 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=1)
       
        x = int(LocationPerFrame[FrameCount][0])
        y = int(LocationPerFrame[FrameCount][1])
        I_Cx= int(math.floor(x/256))
        I_Cy=int(math.floor(y/144))

        
        if PredictFlag:
            ModelLen=len(UserModelList)

            
            Result=list(Predict_result_Top5[I_Cy*N_tile+I_Cx][0])
            Result_W=list(Predict_result_Top5_Score[I_Cy*N_tile+I_Cx][0])
            #print (">>>>>>>>>>>>",len(Result))
            #print (Result)
            for kk in xrange(len(Result)):
                #action = GetActionFromeResult_Single(Predict_result_Top5[i * N_tile + j])
                action = GetActionFromeResult_SingleB(Result[kk])
                
                NotFind=0
                try:
                    K = UserModelList.index(action)
                except:
                    NotFind=1
                else:
                    NotFind=0
                if NotFind:
                    #print("not find")
                    
                    #if ModelLen > ConstModelLen:
                     #   del UserModelList[0]
                    
                    UserModelList.append(action)
                    UserModelListW.append(4)
                else:
                    UserModelListW[K]+=4
                #print(action)
            ModelLen = len(UserModelList)
            Ind_List=[]
            for kkk in xrange(ModelLen):
                UserModelListW[kkk]-=1
            for kkk in xrange(ModelLen):
                if UserModelListW[kkk]<=0:
                    Ind_List.append(kkk)
            F=len(Ind_List)
            for kkk in xrange(F):
                del UserModelListW[Ind_List[F-1-kkk]]
                del UserModelList[Ind_List[F - 1 - kkk]]
          
            max_num_index_list = map(UserModelListW.index, heapq.nlargest(5, UserModelListW))
            
            print(max_num_index_list)
            for kkk in max_num_index_list:
                print(UserModelList[kkk])
            #UserModelList.append()
        # print(">>>>>>>>>>>>>>>>     ",x,y,"   <<<<<<<<<<<<<<<<<<<<<<<")
        
        cv2.circle(Whole_Frame, (x, y), point_size, point_color, thickness)
        cv2.rectangle(Whole_Frame, (x - 50, y - 50), (x + 50, y + 50), (0, 255, 0), 2)
        
        # frame = frame[1*256:2*256+0,1*340+150:2*340+150]
        frame = Frame_List[2]
        # cv2.putText(frame, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=2)
        cv2.imshow('Frames', Whole_Frame)
        # cv2.imshow('Frames', Frame_List[2])

        img = cv2.resize(frame, dims[1::-1])
        last_16_frames.append(img)

        # img to img_List
        # Total NO = N_tile * N_tile
        img_List = []
        
        for i in xrange(N_tile):
            for j in xrange(N_tile):
                # img_List.append(cv2.resize(Frame_List[i*N_tile+j], dims[1::-1]))
                last_16_frames_List[i * N_tile + j].append(cv2.resize(Frame_List[i * N_tile + j], dims[1::-1]))

        if frame_counter == (batch_size * 6):
            frame_counter = 0
        frame_counter = frame_counter + 1
        
        flag_Count = 0
        if (frame_counter % batch_size == 0):
            PredictFlag=1
            Buffer_Count += 1
            for i in range(N_tile):
                for j in range(N_tile):
                    Flag_working[i * N_tile + j] = 1
            for i in range(N_tile):
                for j in range(N_tile):
                    running_frames_T[i * N_tile + j].append(last_16_frames_List[i * N_tile + j])
                    # =================================================================================
                    # predict_ind, top5_label=Prediction_Core(dims,running_frames_T[i * N_tile + j],algo,image_mean,net,num_categories,index_to_label)
                    # Predict_result[i * N_tile + j]=index_to_label[int(predict_ind)]

                    try:
                        # thread.start_new_thread(Prediction_Core_Thread, (
                        # dims, running_frames_T[i * N_tile + j], algo, image_mean, net, num_categories, index_to_label,
                        # Index_l[i * N_tile + j],))

                        thread.start_new_thread(Prediction_Core_Thread, (
                            dims, running_frames_T[i * N_tile + j], algo, image_mean, Caffe_net[j], num_categories,
                            index_to_label,
                            Index_l[i * N_tile + j],))

                        # thread.start_new_thread(print_time, ("Thread-2", 4, 2,))
                    except:
                        print "Error: unable to start thread"
            flagIn = 1
            Flag_W = True
            # while flag_Count<N_tile*N_tile-1:
            while Flag_W == True:
                Flag_W = False
                for i in range(N_tile):
                    for j in range(N_tile):
                        if Flag_working[i * N_tile + j] == 1:
                            Flag_W = True
                flagIn = 0  # pass

            
            print ("========================================================")
            for i in range(N_tile):
                for j in range(N_tile):
                    
                    if i == 2 and j == 2:
                        print(
                        Predict_result_Top5[i * N_tile + j][0][2], Predict_result_Top5_Score[i * N_tile + j][0][2], i,
                        j)
                        print(
                        Predict_result_Top5[i * N_tile + j][0], Predict_result_Top5_Score[i * N_tile + j][0], i, j)
                        action = GetActionFromeResult_Single(Predict_result_Top5[i * N_tile + j])
                        # print ("Action is: ", action)
                        print (action)

            print ("========================================================")

            last_16_frames_List = []
            # clean last_16_frames_List
            for i in range(N_tile):
                for j in range(N_tile):
                    last_16_frames_sub = []
                    last_16_frames_List.append(last_16_frames_sub)
            t1 = tm.time()
            print("Predicted Time:", t1 - t0, "Predicted Id:", Predict_result_Top5[0], "# of buffer", Buffer_Count)
            # text = "Action: " + index_to_label[int(predict_ind)]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if FrameCount> TotalFrames-30:
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    '''===========================
    Work done in 2019 and the author is no longer working on this
    For further discussion please send email to fengx17@miamioh.edu
    ==========================='''
    TestVideoFlow = False
    if TestVideoFlow == True:
        Name = "a"
        i = 0
        FrameRate = 30
        userLoca_One(FrameRate, Name, i)
        exit()

    # model files
    mean_file = "rgb_mean.mat"
    model_def_file = 'ECO_Lite_kinetics.caffemodel'
    model_file = 'deploy.prototxt'
    # class indices file
    classes_file = "class_ind_kinetics.txt"
    # num_categories
    num_categories = 400

    online_predict(mean_file, model_def_file, model_file, classes_file, num_categories)

