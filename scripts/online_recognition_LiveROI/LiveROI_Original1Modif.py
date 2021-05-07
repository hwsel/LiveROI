# -*- coding:utf-8 -*-

'''============================================================
Modification for E:

Debug the multithread

fix action detection result format error: delete "\'" problem
============================================================='''
"""script for predicting labels from live feed"""
import sys

sys.path.append('/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python')
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


def userLoca_One(FrameRate, FileNameStart, i):
    """
    :param FrameRate: FPS
    :param FileNameStart: The filename will be generated with FileNameStart and the "i"
    :param i: the number of the user
    :return: Userdata
    """

    '''
    # test and collect video info 
    video_name = '/home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Video/1-1-Conan Gore FlyB.mp4'
    # video_name = '/home/kora/Downloads/1-7-Cooking Battle.mp4'
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
    # /home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Location
    VideoName = "video_0"
    DirectorName = "/home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Location/"
    i = 1
    # UserFile = FileNameStart + str(i) + ".csv"
    UserFile = DirectorName + VideoName + '_' + str(i) + ".csv"
    Userdata = []
    flagTime = 1
    TimeStamp = []
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到Userdata中
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])
    flagTime = 0
    Userdata = [[float(x) for x in row] for row in Userdata]  # 将数据从string形式转换为float
    Userdata = np.array(Userdata)  # 将list数组转化成array数组便于查看数据结构
    TOT_Len = len(Userdata)
    print(TOT_Len)
    print(Userdata.shape)
    NUmberCount = 0
    while NUmberCount < TOT_Len:
        # NUmberCount+=1
        UserIn1s = []
        UserIn1sR = []
        Ind = 0
        while PreTime > CurTime:
            strA = TimeStamp[j].split(':')
            # print(strA[2])
            # Ind=Ind+1
            # print([Ind,j])
            CurTime = math.floor(float(strA[2]))
            if CurTime == 0 and PreTime == 60:
                break
            UserAll = []
            for k in range(0, A):
                # print(k,j)
                x = ProceList[k][j][1]
                y = ProceList[k][j][2]
                z = ProceList[k][j][3]
                w = ProceList[k][j][4]
                H, W = LocationCalculate(x, y, z, w)
                IH = math.floor(H * FrameHeightT)
                IW = math.floor(W * FrameWidthT)
                UserAll.append([IW, IH])
                # print(IW,IH)
            j = j + 1
            UserIn1s.append(UserAll)
        FrameCount = 0
        PreTime = CurTime + 1


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


def GetActionFromeResult_SingleB(Predict_result_Top5,i):
    # userInput1 = raw_input("Type the phrase1: ")
    # print (userInput1)
    # AA = "woqu"
    # print (AA)
    EE = Predict_result_Top5[0][i]
    IN_K = EE.index('\'')
    CC = EE[:-1]
    # print ("IN_K====>", IN_K)
    BB = CC[IN_K + 1:]
    # print (BB)
    return BB

def userLoca_All():
    ProceList = []
    VideoUserName = "video_3_"
    TimeStamp = []
    flagTime = 1

    '''
    读取用户历史数据 48 名用户  A设定用户数量 A=49来加入48 用户 A=1 只读取一个用户数据
    '''
    A = 1  # 47                    #49
    for i in range(1, A + 1):  # 49
        UserFile = VideoUserName + str(i) + ".csv"
        Userdata = []
        with open(UserFile) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            birth_header = next(csv_reader)  # 读取第一行每一列的标题
            for row in csv_reader:  # 将csv 文件中的数据保存到Userdata中
                Userdata.append(row[1:])
                if flagTime == 1:
                    TimeStamp.append(row[0])
        flagTime = 0
        Userdata = [[float(x) for x in row] for row in Userdata]  # 将数据从string形式转换为float
        Userdata = np.array(Userdata)  # 将list数组转化成array数组便于查看数据结构
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
        while PreTime > CurTime:
            strA = TimeStamp[j].split(':')
            # print(strA[2])
            # Ind=Ind+1
            # print([Ind,j])
            CurTime = math.floor(float(strA[2]))
            if CurTime == 0 and PreTime == 60:
                break
            UserAll = []
            for k in range(0, A):
                # print(k,j)
                x = ProceList[k][j][1]
                y = ProceList[k][j][2]
                z = ProceList[k][j][3]
                w = ProceList[k][j][4]
                H, W = LocationCalculate(x, y, z, w)
                IH = math.floor(H * FrameHeightT)
                IW = math.floor(W * FrameWidthT)
                UserAll.append([IW, IH])
                # print(IW,IH)
            j = j + 1
            UserIn1s.append(UserAll)
        FrameCount = 0
        PreTime = CurTime + 1

        # 加入1s内用户抽样，原用户并不是一秒30个
        if len(UserIn1s) < 30:
            for k in range(0, len(UserIn1s)):
                UserIn1sR.append(UserIn1s[k])
            for k in range(len(UserIn1s) - 1, 30):
                UserIn1sR.append(UserIn1s[len(UserIn1s) - 1])
        else:
            for k in range(0, 30):
                UserIn1sR.append(UserIn1s[k])
            # UserIn1s.append(UserAllR)
            # =========test
            # label[4] = 1
            # label[6] = 1
        label = []
        for i in range(len(UserIn1sR)):
            label.append(check_box(UserIn1sR[i][0][0], UserIn1sR[i][0][1]))

        # label[check_box(UserIn1sR[0][0][0], UserIn1sR[0][0][1])]=1
    else:
        print'=='


# 整理完每一帧的用户视角之后，播放一秒内的每一帧
# ================================

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


'''
# 行为识别的核心模块，此为多线程版本
# 多线程版本中返回结果直接写入数组中
# 多线程版本中输入参数主要添加i和j
使用的数组有 （函数开始的时候完成初始化）
 Predict_result = []  普通列表
Predict_result_Score=[]     普通列表
Predict_result_Top5 = []    类型是列表的列表
Predict_result_Top5_Score = []      类型是列表的列表
控制变量 flag_Count 初始为0,线程中累加
'''


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
    '''
        获取结果后将结果保存到公共数组里面
    '''
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


def online_predict(mean_file, model_def_file, model_file, classes_file, num_categories,V_inde):
    Buffer_Count = 0
    # ===========================================
    KeyWords = "cooking"
    if TestWords == True:
        phraseVector1 = PhraseVector(KeyWords)
    # ============parameters====================
    N_tile = 5
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


    # video_name = '/home/kora/Downloads/2-5-FightingB.mp4'
    # 2-2-VoiceToyB 2-1-KoreanB 1-1-Conan Gore FlyB 2-4-FemaleBasketballB 2-8-reloadedB
    video_name = '/home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Video/1-7-Cooking BattleB.mp4'
    # video_name = '/home/kora/Downloads/1-7-Cooking Battle.mp4'
    video_name_head = '/home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Video/'
    Video_FileList = ["2-8-reloadedB", "2-2-VoiceToyB", "1-1-Conan Gore FlyB", "1-2-FrontB", "1-6-FallujaB",
                      "1-7-Cooking BattleB", "1-8-FootballB",
                      "1-9-RhinosB", "2-1-KoreanB", "2-3-RioVRB", "2-4-FemaleBasketballB", "2-5-FightingB",
                      "2-6-AnittaB"]
    VideoINDX = V_inde
    print( Video_FileList[VideoINDX])
    print(VideoINDX, Video_FileList[12],len(Video_FileList))
    VideoNameFile = video_name_head + Video_FileList[VideoINDX] + '.mp4'
    for k in range(23):
        print(k)
    return True



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

    cap = cv2.VideoCapture(VideoNameFile)
    W_Frame = cap.get(3)
    H_Frame = cap.get(4)
    TotalFrames = cap.get(7)

    """
        save the action result
        """
    fileNameCSV_Save = "00_" + Video_FileList[VideoINDX] + "ACR_Result.csv"
    outAccuBanDRes_Save = open(fileNameCSV_Save, 'a')
    csv_writeABF_Save = csv.writer(outAccuBanDRes_Save, dialect='excel')
    ACCbadRestul = ["Overhead", "Action 1", "Action 2", "Action 3", "Action 4", "Action 5", "Score 1", "Score 2",
                    "Score 3", "Score 4", "Score 5"]
    csv_writeABF_Save.writerow(ACCbadRestul)


    # check the parameters of get frome:  https://blog.csdn.net/qhd1994/article/details/80238707
    print("===============frame width============")
    print (W_Frame)
    print("===============frame height============")
    print (H_Frame)

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
    # last_16_frames 原来用于存放十六个小图构成一个cube，现在每帧有多个小块，每一个小块，每十六帧要存入一个last_16_frames 中，
    # 于是， 创建last_16_frames_List 存放针对每一个小块的十六个小块。每凑够十六次之后要在处理代码部分中重置清零
    last_16_frames_List = []
    for i in range(N_tile):
        for j in range(N_tile):
            last_16_frames_sub = []
            last_16_frames_List.append(last_16_frames_sub)
    FrameCount=0
    while (True):
        # Capture frame-by-frame
        time = "Frame: " + str(frame_counter)
        t0 = tm.time()
        ret, frame = cap.read()
        FrameCount = FrameCount+1
        ret, frame = cap.read()
        FrameCount = FrameCount + 1
        ret, frame = cap.read()
        FrameCount = FrameCount + 1
        ret, frame = cap.read()
        FrameCount = FrameCount + 1
        Whole_Frame = frame.copy()
        # frame = frame[3*256:4*256+0,5*340+150:6*340+150]
        # 每一帧 根据tile的数量分割成子块。Frame_List 用于存放一帧内的每个小块
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
                if TestWords == True:
                    phraseVector2 = PhraseVector(Predict_result[i * N_tile + j])
                    similarityScore = phraseVector1.CosineSimilarity(phraseVector2.vector) * 10
                    print(KeyWords, Predict_result[i * N_tile + j], similarityScore)
                    cv2.putText(Whole_Frame, str(similarityScore), (j * 256 + 10, i * 144 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=1)

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
        # 当进行多线程时，以上代码仅处理一个小块，现在我们通过列表处理没一个小块，首先需要一个列表来存储这些
        # 使用 last_16_frames_List
        for i in xrange(N_tile):
            for j in xrange(N_tile):
                # img_List.append(cv2.resize(Frame_List[i*N_tile+j], dims[1::-1]))
                last_16_frames_List[i * N_tile + j].append(cv2.resize(Frame_List[i * N_tile + j], dims[1::-1]))

        if frame_counter == (batch_size * 6):
            frame_counter = 0
        frame_counter = frame_counter + 1
        '''=================================================================================================
        step 1：  当收集十六帧的数据之后，每一帧中每一块凑齐16个小图后存入列表running_frames_T中。调用Prediction_Core来干活

        step 2：  加入多线程支持  原来的PredictionCore改版，新版直接在函数中写入数组中。
         使用的数组有 （函数开始的时候完成初始化）
         Predict_result = []  普通列表
        Predict_result_Score=[]     普通列表
        Predict_result_Top5 = []    类型是列表的列表
        Predict_result_Top5_Score = []      类型是列表的列表
        控制变量 flag_Count 初始为0,线程中累加
        =================================================================================================='''
        flag_Count = 0
        if (frame_counter % batch_size == 0):
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

            '''
                此处，各线程已经完成对每一块视频的行为识别处理
                技术细节（tile×tile 块， 每两秒16帧）
                接下来的代码块实现对结果的分析。完成基于行为识别的视角预测
                Predict_result_Top5 = []
                Predict_result_Top5_Score = []
            '''
            '''
            print ("========================================================")
            for i in range(N_tile):
                for j in range(N_tile):
                    # 打印出所有的结果检查
                    # print(Predict_result_Top5[i * N_tile + j][0],Predict_result_Top5_Score[i * N_tile + j][0],i,j)
                    # 针对2,2号块进行检查
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
            '''


            last_16_frames_List = []
            # clean last_16_frames_List
            for i in range(N_tile):
                for j in range(N_tile):
                    last_16_frames_sub = []
                    last_16_frames_List.append(last_16_frames_sub)
            t1 = tm.time()
            print("Predicted Time:", t1 - t0, "Predicted Id:", Predict_result_Top5[0], "# of buffer", Buffer_Count)


            for i in range(N_tile):
                for j in range(N_tile):
                    ACtionList=[]
                    ACtionList.append(str(t1 - t0))
                    for KK in range(5):
                        action = GetActionFromeResult_SingleB(Predict_result_Top5[i * N_tile + j],KK)
                        ACtionList.append(action)
                        #print(">>>>>>>>>>>>>>>>>>>>>>>>",action)
                    ACtionList.append(Predict_result_Top5_Score[i * N_tile + j][0][0])
                    ACtionList.append(Predict_result_Top5_Score[i * N_tile + j][0][1])
                    ACtionList.append(Predict_result_Top5_Score[i * N_tile + j][0][2])
                    ACtionList.append(Predict_result_Top5_Score[i * N_tile + j][0][3])
                    ACtionList.append(Predict_result_Top5_Score[i * N_tile + j][0][4])
                    #print(")))))))))))))))",ACtionList)

                    csv_writeABF_Save.writerow(ACtionList)

            # text = "Action: " + index_to_label[int(predict_ind)]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if FrameCount > TotalFrames - 30:
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
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
    for V_inde in range(0,14,1):
        print(V_inde)
        online_predict(mean_file, model_def_file, model_file, classes_file, num_categories,V_inde)
    #exit()


