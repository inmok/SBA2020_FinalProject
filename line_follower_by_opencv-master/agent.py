#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf

class CNN:
    
    def __init__(self, h, w, sess):
        self.size_h = h
        self.size_w = w
        self.sess = sess
        self.model = self.make_model()
        
    def make_model(self):
        
        # žðµš
        self.observation = tf.placeholder(shape=[None, self.size_h, self.size_w, 1], dtype=tf.float32) # ÀÌ¹ÌÁö µ¥ÀÌÅÍ
        self.label = tf.placeholder(shape=[None, 4], dtype=tf.int32) # ¶óº§ µ¥ÀÌÅÍ
        # L1 ImgIn shape=(?,self.size_h, self.size_w,1)
        self.w_in = tf.Variable(tf.random_normal([3,3,1,10], stddev=.05)) # conv 1 °¡ÁßÄ¡
        self.l1 = tf.nn.conv2d(self.observation, self.w_in, strides=[1,1,1,1], padding='SAME')
        #    Conv     -> (?, self.size_h, self.size_w, 10)
        self.l1 = tf.nn.relu(self.l1)
        self.l1 = tf.nn.max_pool(self.l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        #    Pool     -> (?, self.size_h/2, self.size_w/2, 10)
        
        
        ###
        # L2 ImgIn shape=(?,self.size_h/2, self.size_w/2,10)
        self.w2_in = tf.Variable(tf.random_normal([3,3,10,40], stddev=.01)) # conv 2 °¡ÁßÄ¡
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
        print('> žðµš »ýŒº ¿Ï·á')
        
        
    def train(self, batch_in, batch_label):
        # ÇÐœÀœÃÇà.
        _, cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.observation:batch_in, self.label:batch_label})
        return cost
    def policy(self, input_data):
        # policy žž µµÃâÇÏ±â
        output = self.sess.run(self.output, feed_dict={self.observation:input_data})
        return output
    def test(self, in_test, label_test):
        # ±æÀ» Àß ÀÎœÄÇÏŽÂÁö Å×œºÆ®.
        # Å×œºÆ® µ¥ÀÌÅÍ ŒÂÀ» ¹Ìž® ÀúÀåÇØ³õ°í °ËÁõÇÏŽÂ žÞŒ­µå
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
        # œÇœÃ°£Àž·Î ±æÀ» Àß ÀÎœÄÇÏŽÂÁö Å×œºÆ®ÇÏŽÂ žÞŒ­µå.
        out = self.sess.run(self.output, feed_dict={self.observation:d})