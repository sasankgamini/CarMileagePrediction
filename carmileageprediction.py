import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
data=pd.read_csv('auto-mpg.csv')
##print(data.head())
a=data['weight']
b=data['acceleration']
c=data['displacement']
mileage=data['mpg']
newa=list(a)
newb=list(b)
newc=list(c)
newmileage=list(mileage)
newa=preprocessing.minmax_scale(newa)
newb=preprocessing.minmax_scale(newb)
newc=preprocessing.minmax_scale(newc)
correctmileage=tf.placeholder(tf.float32)
weight=tf.placeholder(tf.float32)
acceleration=tf.placeholder(tf.float32)
displacement=tf.placeholder(tf.float32)
p=tf.Variable(1.2,tf.float32)
q=tf.Variable(1.3,tf.float32)
r=tf.Variable(1.1,tf.float32)
s=tf.Variable(1.4,tf.float32)
learningrate=0.0001
predictedoutput=p*weight+q*acceleration+r*displacement+s
squared_delta=tf.square(predictedoutput-correctmileage)
loss=tf.reduce_sum(squared_delta)
optimizer=tf.train.AdamOptimizer(learningrate)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for n in range(1,2000,1):
    output=sess.run(loss,{weight:newa,acceleration:newb,displacement:newc,correctmileage:newmileage})
    sess.run(train,{weight:newa,acceleration:newb,displacement:newc,correctmileage:newmileage})
pvalue=sess.run(p)
qvalue=sess.run(q)
rvalue=sess.run(r)
svalue=sess.run(s)
print(output)
print(pvalue)
print(qvalue)
print(rvalue)
print(svalue)
sess.close()
    
    
