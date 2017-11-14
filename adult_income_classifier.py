'''
Tensorflow Neural Network Test Program, by Kevin Rong & Kirsten Gillam
Data used from https://archive.ics.uci.edu/ml/datasets/Adult
Simple neural network that predicts whether a person has a annual salary of >50K or <=50K
based on their age, education level, occupation, sex and hours per week.
Contains two hidden layers of 6/3 neurons and gives a single output.
Hidden layers use leaky ReLU and the output uses the sigmoid function.
The best accuracy I got was 81.05%.

This is my first program in Python, so there's some bad programming practice, sorry in advance.
'''

import tensorflow as tf
import random


def get_data(filename):
    # inputData=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    inputData = []
    inputResult = []
    input_file = open(filename, "r")
    '''
    age
    education
    occupation(14)
    sex
    hours-per-week
    '''

    def occupationConverter(name):
        # , , , , , , , , , , , , ,
        if (name == " Tech-support"):
            return 0
        if (name == " Craft-repair"):
            return 1
        if (name == " Other-service"):
            return 2
        if (name == " Sales"):
            return 3
        if (name == " Exec-managerial"):
            return 4
        if (name == " Prof-specialty"):
            return 5
        if (name == " Handlers-cleaners"):
            return 6
        if (name == " Machine-op-inspct"):
            return 7
        if (name == " Adm-clerical"):
            return 8
        if (name == " Farming-fishing"):
            return 9
        if (name == " Transport-moving"):
            return 10
        if (name == " Priv-house-serv"):
            return 11
        if (name == " Protective-serv"):
            return 12
        if (name == " Armed-Forces"):
            return 13
        return -1

    def Lerp(value, min, max):
        return (value - min) / (max - min)

    ageMin = 9999
    ageMax = -1
    hrswkMin = 9999
    hrswkMax = -1
    for line in input_file:
        if len(line) < 25:
            continue
        if '?' in line.split(',')[0]:
            continue
        age = int(line.split(',')[0])
        if '?' in line.split(',')[4]:
            continue
        education_type = int(line.split(',')[4])
        occupation_type = occupationConverter(line.split(',')[6])
        if occupation_type == -1:
            continue
        if '?' in line.split(',')[9]:
            continue
        sex = -1
        if line.split(',')[9] == ' Male':
            sex = 0
        else:
            sex = 1
        if '?' in line.split(',')[12]:
            continue
        hoursPerWeek = int(line.split(',')[12])
        if '?' in line.split(',')[14]:
            continue
        result = -1
        if "<=50" in line.split(',')[14]:
            result = 0
        else:
            result = 1
        tmpInput = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tmpInput[0] = age
        tmpInput[1] = education_type
        tmpInput[2 + occupation_type] = 1
        tmpInput[16] = sex
        tmpInput[17] = hoursPerWeek
        inputData.append(tmpInput)
        inputResult.append([result])
        if (ageMax < age):
            ageMax = age
        if (ageMin > age):
            ageMin = age
        if (hrswkMax < hoursPerWeek):
            hrswkMax = hoursPerWeek
        if (hrswkMin > hoursPerWeek):
            hrswkMin = hoursPerWeek
    for i in range(len(inputData)):
        inputData[i][0] = Lerp(inputData[i][0], ageMin, ageMax)
        inputData[i][1] = Lerp(inputData[i][1], 1, 16)
        inputData[i][17] = Lerp(inputData[i][17], hrswkMin, hrswkMax)
    # print("DEBUG ageMin=",ageMin,", ageMax=",ageMax,", hrswkMin=",hrswkMin,", hrswkMax=",hrswkMax,", data=",inputData[0])
    input_file.close()
    return inputData, inputResult


trainData, trainResults = get_data("adult.data")
print("Got", len(trainData), "train data")
testData, testResults = get_data("adult.test")
print("Got", len(testData), "test data")

placeholderInput = tf.placeholder(tf.float32, [None, 18])
weightsL1 = tf.Variable(tf.random_normal([18, 6]))
biasesL1 = tf.Variable(tf.random_normal([6]))
weightsL2 = tf.Variable(tf.random_normal([6, 3]))
biasesL2 = tf.Variable(tf.random_normal([3]))
weightsL3 = tf.Variable(tf.random_normal([3, 1]))
biasesL3 = tf.Variable(tf.random_normal([1]))

outputL1 = tf.nn.leaky_relu(tf.matmul(placeholderInput, weightsL1) + biasesL1)
outputL2 = tf.nn.leaky_relu(tf.matmul(outputL1, weightsL2) + biasesL2)
networkOutput = tf.sigmoid(tf.matmul(outputL2, weightsL3) + biasesL3)

placeholderAnswer = tf.placeholder(tf.float32, [None, 1])
cost = tf.reduce_mean(tf.square(networkOutput - placeholderAnswer))
trainer = tf.train.AdamOptimizer(0.1).minimize(cost)

isPredictionCorrect = tf.less(tf.abs(networkOutput - placeholderAnswer), 0.5)
accuracy = tf.reduce_mean(tf.cast(isPredictionCorrect, tf.float32))
accuratePredictions = tf.reduce_sum(tf.cast(isPredictionCorrect, tf.float32))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()
random.seed();
maxAccuracyIter = -1
maxAccuracy = session.run(accuracy, feed_dict={placeholderInput: testData, placeholderAnswer: testResults})
print("Before training cost=",
      session.run(cost, feed_dict={placeholderInput: testData, placeholderAnswer: testResults}), "accuracy=",
      maxAccuracy)
for iterations in range(2000):
    slice = random.sample(range(len(trainData)), 1000)
    trainDataSample = []
    trainResultsSample = []
    for i in slice:
        trainDataSample.append(trainData[i])
        trainResultsSample.append(trainResults[i])
    session.run(trainer, feed_dict={placeholderInput: trainDataSample,
                                    placeholderAnswer: trainResultsSample})
    curAccuracy = session.run(accuracy, feed_dict={placeholderInput: testData, placeholderAnswer: testResults})
    print("Train iteration", iterations, "cost=",
          session.run(cost, feed_dict={placeholderInput: trainDataSample, placeholderAnswer: trainResultsSample}),
          "accuracy=",
          curAccuracy)
    if (curAccuracy > maxAccuracy):
        maxAccuracy = curAccuracy
        maxAccuracyIter = iterations
print("Done, max acc is", maxAccuracy, "at iteration", maxAccuracyIter)
