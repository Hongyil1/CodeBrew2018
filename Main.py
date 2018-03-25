"""
Code for CodeBrew2018
University of Melbourne
Team NoDemic
2018.03.25
"""

from __future__ import print_function
import os
import csv
import smtplib
import tensorflow as tf
import numpy as np
from pymongo import MongoClient
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# read data from excel file:
def read_data():
    """
    read data from csv file and return set X and Y
    x_data: [value, baseline]
    y_data: [1, 0, 0] for "Same", [0, 1, 0] for "Better", [0, 0, 1] for "Worse"
    """
    file_name = "data.csv"
    with open(file_name, "r") as csvfile:
        file = csv.reader(csvfile)
        # transpose to make it easier to iterate over
        fileT = list(zip(*file))
        case = [float(x) for x in fileT[1][1:]]

    allBaselines = []
    allTags = []

    for i in range(len(case)):
        # if 10 baselines already recorded, overwrite the oldest baseline
        if i > 9:
            baseline = sum(case[i-9:i+1]) / 10
            remainder = 1 % 10
            # baselines[remainder] = baseline
        # if baselines list not yet full, append baseline to end of list
        else:
            baseline = sum(case[:i+1]) / (i+1)
            # baselines.append(round(baseline, 5))

        ratio = list(reversed(case))[i] / baseline
        # print(list(reversed(case))[i], baseline, ratio)
        tag = ratio_judge(ratio)
        allBaselines.append(round(baseline, 5))
        rev_base = list(reversed(allBaselines))
        allTags.append(tag)
        y_data = list(reversed(allTags))
        x_data = []

    for value, baseline in zip(case, rev_base):
        x_data.append([value, baseline])

    return x_data, y_data

def ratio_judge(ratio):
    # print(ratio)
    if ratio >= 0.95 and ratio <= 1.05:
        # print("aa")
        return [1, 0, 0]
    elif ratio > 1.05:
        # print("bb")
        return [0, 0, 1]
    elif ratio < 0.95:
        # print("cc")
        return [0, 1, 0]

def add_layer(inputs, in_size, out_size, act_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

    # Decide whther use active function
    if act_function is None:
        # linear function
        outputs = Wx_plus_b
    else:
        outputs = act_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = {xs: v_xs, ys: v_ys})
    return result

def retrieveMostRecent(noRecords):
    # , sort = {'Date', -1}))#.limit(noRecords).sort('Date',-1))
    return (collection.find(limit=noRecords).sort('Date', -1))

def printMostRecent(noRecords):
    last10 = []
    for doc in retrieveMostRecent(noRecords):
        last10.append(doc)
    return last10

def data_process(last10):
    values = []
    for value in last10:
        Venezuela = value['Venezuela']
        values.append(Venezuela)
    latest = values[0]
    baseline = sum(values) / len(values)

    return [latest, baseline]

def indext_to_tag(index):
    if index == 0:
        return "Same"
    elif index == 1:
        return "Better"
    elif index == 2:
        return "Worse"

def send_emial(my_add, my_pw, send_add, Subject, body):
    fromaddr = my_add
    toaddr = send_add
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = Subject

    body = body
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, my_pw)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    print("Email has been sent")


if __name__=="__main__":
    x_data, y_data = read_data()
    x_array = np.asarray(x_data)
    y_array = np.asarray(y_data)

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 2])  # input size 2
    ys = tf.placeholder(tf.float32, [None, 3])  # output size 1

    # hidden layer
    l1 = add_layer(xs, 2, 10, act_function=tf.nn.relu)

    # output layer
    prediction = add_layer(l1, 10, 3, act_function=tf.nn.softmax)

    # # error between prediction and real data
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
    #                      reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys* tf.log(prediction),
                                                  reduction_indices=[1]))

    # Training
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Initilize all the data
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    sess = tf.Session()
    # with tf.Session() as sess:
    sess.run(init)
    # train 2000 times
    for i in range(1001):
        sess.run(train_step, feed_dict={xs: x_array, ys: y_array})
        if i % 50 == 0:
            accuracy = compute_accuracy(x_array, y_array)
            print("Number of training: ",i, " Accuracy: ",accuracy)
    save_path = saver.save(sess, save_path="/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

    # Read database
    URI = 'mongodb://HHATK:CBrew2018@ds123499.mlab.com:23499/cbrew2018'
    connection = MongoClient(URI)
    db = connection['cbrew2018']
    collection = db['recordsnew']
    last10 = printMostRecent(10)

    # Process the data and print the prediction
    new_X = data_process(last10)
    x_newarray = np.asarray([new_X])
    print("New Instance: ", x_newarray)
    pred_y = sess.run(prediction, feed_dict={xs: x_newarray})
    index = np.argmax(pred_y)
    tag = indext_to_tag(index)
    print("Prediction Result: ", tag)

    # Send the email
    my_add = "testcodebrew2018@gmail.com"
    my_pw = "CodeBrew2018"
    send_add = "testcodebrew2018@gmail.com"
    Subject = "CodeBrew2018"
    if(tag == "Better"):
        body = "Hi, \n\nGreat news, the situation is improving!\n\nGreat Job,\nNoDemic Team"
    elif(tag == "Worse"):
        body = "Hi, \n\nThe situation is worsening, please review and respond appropriately.\n\nThanks, \nNoDemic Team"
    else:
        body = "Hi,\n\nThe situation hasn't changed, we'll update you again tomorrow!\n\nRegards,\nNoDemic Team"
    send_emial(my_add, my_pw, send_add, Subject, body)