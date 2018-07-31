import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from math import isnan


def load_traindata(item_del,item_con,item_label,item_modify):
    #载入训练数据函数，输入参数分别为训练中不需要的列名，需要进行转化的列名以及标签列名，还有需要进行特殊处理的列名
    data = pd.read_csv("train.csv",header=0)
    ind = data.keys()
    ind_del = []
    temp = data
    temp = temp.fillna(100)
    #将原始数据中缺失的值替换成100
    for j in item_del:
        ind_del.append(np.where(ind==j)[0].item())
        del(temp[j])
    #删除抛弃的列
    m,n = temp.shape
    #记录删除之后的数据矩阵大小
    label = temp[item_label].values
    del(temp[item_label[0]])
    #获得训练数据的标签，并且在原始数据中将其剔除
    ind_temp = temp.keys()
    ind_mod = np.where(ind_temp == item_modify[0])
    #获得需要进行处理的列索引
    temp2 = temp.values


    for item in item_con:
        sto = []
        ind = np.where(temp.keys() == item)
        for i in range(m):
            if temp[item][i] == 100:
                continue
            elif temp[item][i] not in sto:
                sto.append(temp[item][i])
            temp2[i,ind] = sto.index(temp[item][i])

    #将原始数据中的字符信息转换成标量信息
    return temp2,label,ind_mod
    #返回训练数据，标签以及需要处理数据的列索引

def modify(data,ind_mod,thresh1,thresh2):
    #输入参数为原始数据，需要处理数据的列索引，阈值1，阈值2
    m,n = data.shape
    ind_P = 4
    #将该列信息作为参考，对缺失数据进行补充
    #重新将年龄信息进行划分，thresh1以下的视为青少年，thresh2以上的视为老年人

    for j in ind_mod:
        for i in range(m):
            if data[i,j] == 100:
                if data[i,ind_P] == 0: data[i,j] = 1
                else: data[i,j] = 0
                continue
            elif data[i,j] < thresh1: data[i,j] = 0
            elif data[i,j] < thresh2: data[i,j] = 1
            else: data[i,j] = 2

    return data


def load_testdata(item_del,item_con,item_modify):
    #读取训练数据参数，输入参数为要删除的列名，要转换的列名，要进行处理的列名
    data = pd.read_csv("test.csv",header=0)
    ind = data.keys()
    ind_del = []
    temp = data
    temp = temp.fillna(100)
    for j in item_del:
        ind_del.append(np.where(ind==j)[0].item())
        del(temp[j])
    #删除制定列
    m,n = temp.shape
    ind_temp = temp.keys()
    ind_mod = np.where(ind_temp == item_modify[0])
    temp2 = temp.values


    for item in item_con:
        sto = []
        ind = np.where(temp.keys() == item)
        for i in range(m):
            if temp[item][i] == 100:
                continue
            elif temp[item][i] not in sto:
                sto.append(temp[item][i])
            temp2[i,ind] = sto.index(temp[item][i])
    #转换制定列
    #返回测试数据以及需要进一步处理的数据索引
    return temp2,ind_mod
def load_testlabel():
    #读取测试数据标签数据
    data = pd.read_csv("gender_submission.csv")
    label = data['Survived'].values
    return label
def pluse(data,x):
    #为避免数据中出现零项，加一个常数
    m,n = len(data),len(data[0])
    for i in range(m):
        for j in range(n):
            data[i][j] += x
    return data
def train_net(data,label,data_test,label_test,num_itera=1000,step=0.1):
    #搭建神经网络进行训练
    sess = tf.InteractiveSession()
    num_node = len(data[0])
    x = tf.placeholder(tf.float32,[None,num_node])
    y_ = tf.placeholder(tf.float32,[None,2])
    temp = []
    temp_test = []
    temp_datatest = []
    l = len(label)
    for item in label:
        temp.append(item[0])
    for item in label_test:
        temp_test.append(item)
    # for item in data_test:
    #     temp_datatest.append(item[0])
    Y = tf.one_hot(indices=temp,depth=2,on_value=1,off_value=0,axis=1).eval()
    # Y_test = tf.one_hot(indices=label_test,depth=2,on_value=1,off_value=0,axis=1).eval()
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    data_test = scaler.fit_transform(data_test)


    w = tf.Variable(tf.zeros([num_node,2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(x,w)+b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(step).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_itera):
        sess.run(train_step,feed_dict={x:data,y_:Y})
        if i%50 == 0:
            print(sess.run(accuracy,feed_dict={x:data,y_:Y}))
    res = sess.run(tf.arg_max(y,1),feed_dict={x:data_test})
    header = ['PassengerId','Survived']
    l = len(res)
    id_begin = 892
    result = []
    id = []
    for i in range(l):
        id.append(id_begin+i)
    for i in range(l):
        temp2 = []
        temp2.append(id[i])
        temp2.append(res[i])
        result.append(temp2)
    df = pd.DataFrame(result,columns=header)
    df.to_csv("res.csv",encoding="utf-8",index=0)

def modify_fare(data):
    ind_fare = -1
    l = len(data)
    for i in range(l):
        if data[i][ind_fare] <20: data[i][ind_fare] = 0
        elif data[i][ind_fare] < 100: data[i][ind_fare] = 1
        else: data[i][ind_fare] = 2
    return data
def data_analysis(data,index,num):
    index_P = 4
    temp = data[:,index]
    temp_P = data[:,index_P]
    l = len(temp)
    temp2 = []
    for item in temp:
        temp2.append(item)
    while 100 in temp2:
        temp2.remove(100)
    stat = {}
    for i in range(l):
        if temp[i] == 100: continue
        else:
            stat.setdefault((temp_P[i],temp[i]),0)
            stat[(temp_P[i],temp[i])]+=1



    plt.figure()
    plt.hist(temp2)
    plt.show()
    # plt.figure()
    # plt.hist(stat.values())
    # plt.show()
    print(stat)


