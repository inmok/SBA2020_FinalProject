#!/usr/bin/env python
# coding: utf-8

# # 로컬에서 학습시키는 스크립트

# In[1]:


import cv2
import tensorflow as tf
import h5py
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# LABEL 4개 짜리 CNN
class CNN:
    
    def __init__(self, h, w, sess):
        self.size_h = h
        self.size_w = w
        self.sess = sess
        self.model = self.make_model()
        
    def make_model(self):
        
        # 모델
        self.observation = tf.placeholder(shape=[None, self.size_h, self.size_w, 1], dtype=tf.float32) # 이미지 데이터
        self.label = tf.placeholder(shape=[None, 4], dtype=tf.int32) # 라벨 데이터
        # L1 ImgIn shape=(?,self.size_h, self.size_w,1)
        self.w_in = tf.Variable(tf.random_normal([3,3,1,10], stddev=.05)) # conv 1 가중치
        self.l1 = tf.nn.conv2d(self.observation, self.w_in, strides=[1,1,1,1], padding='SAME')
        #    Conv     -> (?, self.size_h, self.size_w, 10)
        self.l1 = tf.nn.relu(self.l1)
        self.l1 = tf.nn.max_pool(self.l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        #    Pool     -> (?, self.size_h/2, self.size_w/2, 10)
        
        ###
        # L2 ImgIn shape=(?,self.size_h/2, self.size_w/2,10)
        self.w2_in = tf.Variable(tf.random_normal([3,3,10,40], stddev=.01)) # conv 2 가중치
        self.l2 = tf.nn.conv2d(self.l1, self.w2_in, strides=[1,1,1,1], padding='SAME')
        #    Conv     -> (?, self.size_h/2, self.size_w/2, 40)
        self.l2 = tf.nn.relu(self.l2)
        self.l2 = tf.nn.max_pool(self.l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        #    Pool     -> (?, self.size_h/4, self.size_w/4, 40)
        
        
        self.w_out = tf.Variable(tf.random_normal(shape=[self.size_w//4*self.size_h//4*40, 4], stddev=0.01))
        self.b = tf.Variable(tf.random_normal([4]))
        self.h_flat = tf.reshape(self.l2, [-1, self.size_w//4*self.size_h//4*40])
        self.output = tf.matmul(self.h_flat, self.w_out) + self.b
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=.00005).minimize(self.cost)
        print('> 모델 생성 완료')
        
        
    def train(self, batch_in, batch_label):
        # 학습시행.
        _, cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.observation:batch_in, self.label:batch_label})
        return cost
    def policy(self, input_data):
        # policy 만 도출하기
        output = self.sess.run(self.output, feed_dict={self.observation:input_data})
        return output
    def test(self, in_test, label_test):
        # 길을 잘 인식하는지 테스트.
        # 테스트 데이터 셋을 미리 저장해놓고 검증하는 메서드
        in_test_copied = copy.deepcopy(in_test)
        cnt = 0
        for step in range(len(in_test_copied)):
            key = cv2.waitKey(10) & 0xFF
            if(key == ord('q')):
                break
            out = self.sess.run(self.output, feed_dict={self.observation:in_test_copied[step:step+1]})
            idx = np.argmax(out, axis=1)
            one_hot = np.zeros_like(out)
            one_hot[0][idx[0]] = 1
            '''if(np.all(one_hot == label_test[])):
                cnt += 1'''
            cv2.putText(in_test_copied[step], 'command : {}'.format(one_hot[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX, .4, (0,0,255), 1)
            cv2.imshow('testing..', in_test_copied[step])
        cv2.destroyAllWindows()
        accuracy = cnt / len(in_test)
        print('acc :', accuracy)
    def test_live(self):
        # 실시간으로 길을 잘 인식하는지 테스트하는 메서드.
        out = self.sess.run(self.output, feed_dict={self.observation:d})


#  # test 데이터와 validation 데이터로 나눔 
#  

# In[4]:


with h5py.File('dataset.h5', 'r') as f: # binary 파일로 학습데이터 다루기
    batch_in = np.array(f['observation'])
    #batch_in = np.expand_dims(batch_in, axis=3)
    
    print(batch_in.shape)
    print(batch_in.dtype)
    batch_in.dtype = np.uint8
    
    validation_in = batch_in[ 6 * int(batch_in.shape[0]/10) : 7 * int(batch_in.shape[0]/10)]
    batch_in = list(batch_in[:6 * int(batch_in.shape[0]/10)]) + list(batch_in[7 * int(batch_in.shape[0]/10) : ])
    batch_in = np.array(batch_in)
    
    batch_label = np.array(f['label'])
        
    validation_label = batch_label[ 6 * int(batch_label.shape[0]/10) : 7 * int(batch_label.shape[0]/10)]
    batch_label = list(batch_label[:6 * int(batch_label.shape[0]/10)]) + list(batch_label[7 * int(batch_label.shape[0]/10) : ])
    batch_label = np.array(batch_label)
    
    
    #batch_label = np.array(batch_label[  8 * int(batch_label.shape[0]/10) : ])
    print('> 사전학습 데이터 load & setting 완료')
    
    print(validation_in.shape)
    print(validation_label.shape)
    
    print(batch_in.shape)
    print(batch_label.shape)


# ## 추가 데이터 확인

# In[ ]:


with h5py.File('dataset_2.h5', 'r') as f:
    batch_in_new = np.array(f['observation'])
    batch_in_new = np.expand_dims(batch_in_new, axis=3)
    
    batch_label_new = np.array(f['label'])
    batch_in_new.dtype = np.uint8
    print('> 추가학습 데이터 {} frame, load & setting 완료'.format(len(batch_in_new)))
    print(batch_in_new.shape)
    #batch_in_new = list(batch_in_new[:108]) + list(batch_in_new[113:])
    #batch_label_new = list(batch_label_new[:108]) + list(batch_label_new[113:])
    print('> 편집 후 frame : ', len(batch_in_new))


# In[ ]:


with h5py.File('dataset_3.h5', 'r') as f:
    batch_in_new2 = np.array(f['observation'])
    batch_in_new2 = np.expand_dims(batch_in_new2, axis=3)
    batch_label_new2 = np.array(f['label'])
    batch_in_new2.dtype = np.uint8
    print('> 추가학습 데이터 {} frame, load & setting 완료'.format(len(batch_in_new2)))
    print(batch_in_new2.shape)
    #batch_in_new = list(batch_in_new[:108]) + list(batch_in_new[113:])
    #batch_label_new = list(batch_label_new[:108]) + list(batch_label_new[113:])
    print('> 편집 후 frame : ', len(batch_in_new2))


# In[ ]:


with h5py.File('dataset_4.h5', 'r') as f:
    batch_in_new3 = np.array(f['observation'])
    batch_in_new3 = np.expand_dims(batch_in_new3, axis=3)
    batch_label_new3 = np.array(f['label'])
    #batch_in_new2.dtype = np.uint8
    print('> 추가학습 데이터 {} frame, load & setting 완료'.format(len(batch_in_new3)))
    print(batch_in_new3.shape)
    #batch_in_new = list(batch_in_new[:108]) + list(batch_in_new[113:])
    #batch_label_new = list(batch_label_new[:108]) + list(batch_label_new[113:])
    print('> 편집 후 frame : ', len(batch_in_new3))


# ## [데이터 보강 - 데이터 합칠 때만 사용]
# 

# In[ ]:


batch_in_aggre = list(batch_in) + list(batch_in_new) + list(batch_in_new2) 
batch_label_aggre = list(batch_label) + list(batch_label_new) + list(batch_label_new2) 
batch_in = np.array(batch_in_aggre)
batch_label = np.array(batch_label_aggre)

#batch_in.dtype = np.uint8
print(batch_in.shape)
print('> 최종 데이터 {} frame, 생성 완료'.format(len(batch_in)))
#print(list(batch_in[10]), batch_label[10])


# ## [추가된 데이터를 그대로 파일에 저장 - 필요할 때만]
# 

# In[ ]:


with h5py.File('dataset.h5', 'r') as f:
    del(f['observation'])
    del(f['label'])
    f['observation'] = batch_in_aggre
    f['label'] = batch_label_aggre
    print('> 데이터 추가하여 {} frame, 저장 완료'.format(len(batch_in_aggre)))


# ## [데이터 확인 및 디버깅]

# In[ ]:


for idx in range(len(batch_in)):
    #print(batch_label[idx+1016])
    cv2.putText(batch_in[idx], str(batch_label[idx]) + ' ' + str(idx), (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('s', batch_in[idx])
    if(cv2.waitKey(0) == ord('q')):
        break

cv2.destroyAllWindows()


# # 에포크 반복 방식

# In[ ]:


acc_Array = []
cost_Array = []

width = 320
height = 640
total_epoch = 160
batch_size = 150
validation_batch_size = int(batch_size/10)
with tf.Session() as sess:
    score = 0
    start = time.time()
    model = CNN(width, height, sess)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('> 데이터 학습 중..')
    for epoch in range(total_epoch):
        start_epoch = time.time()
        for i in range(int((batch_in.shape[0]) / batch_size)):
            cost = model.train(batch_in[i*batch_size:(i+1)*batch_size] , batch_label[i*batch_size:(i+1)*batch_size])
            print(cost)
            
        #cost = model.train(batch_in[(i+1)*batch_size:] , batch_label[(i+1)*batch_size:])
        cost_Array.append(cost)
        print('{}/{} epoch, 손실크기 : {}, {} 초 소요'.format(epoch+1, total_epoch, cost, time.time()-start_epoch))
        
        # ACC 
        for i in range(len(validation_in)):   
            score = score + np.equal( np.argmax(model.policy(np.reshape(validation_in[i],(1,320,640,1)))), np.argmax(validation_label[i]))    
        acc = score / len(validation_in)
        print('{}/{} epoch, Accuracy : {}, {} 초 소요'.format(epoch+1, total_epoch, acc, time.time()-start_epoch)) 
        score = 0
        acc_Array.append(acc)
        
        plt.ylim(0,1.5)
        plt.plot(acc_Array, label = "ACC")
        plt.plot(cost_Array , label = "cost")
        plt.legend()
        plt.xlabel('apoch')
        plt.ylabel('acc / cost')
        plt.show()
    
    print('{}/{} epoch, Accuracy : {}, {} 초 소요'.format(epoch+1, total_epoch, acc, time.time()))  
    save_path = 'C:\\model'
    saver.save(sess, save_path) # 모델 저장
    took_time = time.time() - start
    print('> 학습완료, {} 초 소요, 모델의 파라미터가 {} 에 저장됨.'.format(took_time, save_path))
    


# In[ ]:


width = 320
height = 640
save_path = 'C:\\model'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model = CNN(width, height, sess)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path) 
    
    for idx, val in enumerate(validation_in):
        out = model.policy(np.reshape(val, (1,320,640,1)))
        
        accuracy = np.equal( np.argmax(out), np.argmax(validation_label[idx]) ) / (idx + 1)
        print(np.argmax(out),np.argmax(validation_label[idx]))
    print("accuracy",accuracy)

