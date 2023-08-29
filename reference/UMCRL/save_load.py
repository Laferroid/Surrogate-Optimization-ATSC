# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:36:20 2021

@author: hty22
"""
#导入tensorflow1.x
import tensorflow.compat.v1 as tf
import pickle as pkl

#保存模型，可以输入name=新文件名防止覆盖
def save(model,name = 'saved'):
    save_path = './model/'+name
    saver = tf.train.Saver()
    saver.save(model.sess,save_path)
    
#恢复模型，可以输入name=新文件名防止覆盖
def load(model,name = 'saved'):
    save_path = './model/'+name
    saver = tf.train.Saver()
    saver.restor(model.sess,save_path)

#保存训练所需要的状态，动作及奖励用于与训练，可以输入name=新文件名防止覆盖
#注意RL.ep_x貌似是每个episode重置？此策略下最好保存你觉得最好的几个episode
def save_memory(model,name = 'training_data'):
    f = open('./data/'+name+'.pickle','a')
    training_data = {'state':model.ep_obs,'action':model.ep_as, 'reward':model.ep_rs}
    pkl.dump(training_data,f)
    f.close()
    print('training_data saved')
 
#加载训练所需要的状态，动作及奖励用于与训练，可以输入name=新文件名防止覆盖
#注意此处有与训练选项，可以等所有加载完了做
def load_memory(model, name = 'training_data',pre_train = False):
    f = open('./data/'+name+'.pickle')
    training_data = pkl.load(f)
    f.close()
    model.ep_obs.append(training_data['state'])
    model.ep_as.append(training_data['action'])
    model.ep_rs.append(training_data['reward'])
    if pre_train:
        for i in range(10):
            model.learn()
