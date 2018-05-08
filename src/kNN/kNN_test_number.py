import numpy as np
import matplotlib.pyplot as plt
import operator
import os


def img2vector(file):
    vec = np.zeros((1,1024))
    with open(file) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                vec[0,32*i+j] = int(line[j])
    return vec

def knn(test_data, train_data, level_set, k):
    distances = (((np.tile(test_data, (train_data.shape[0], 1)) - train_data) ** 2).sum(axis=1)) ** 0.5
    sorted_distance = distances.argsort()
    class_count = {}
    for i in range(k):
        level = level_set[sorted_distance[i]]
        class_count[level] = class_count.get(level, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def read_testdata(folder):
    labels = []
    files = os.listdir(folder)
    m = len(files)
    matrix = np.zeros((m,1024))
    for i in range(m):
        file_name = files[i].split('.')[0]
        class_number = file_name.split('_')[0]
        labels.append(class_number)
        matrix[i,:] = img2vector(file=folder+'/'+files[i])
    return labels,matrix

def main():
    train_labels = []
    train_files = os.listdir('data/trainingDigits')
    m = len(train_files)
    train_matrix = np.zeros((m, 1024))
    for i in range(m):
        file_name = train_files[i].split('.')[0]
        class_number = file_name.split('_')[0]
        train_labels.append(class_number)
        train_matrix[i,:] = img2vector(file='data/trainingDigits'+'/'+train_files[i])

    test_files = os.listdir('data/testDigits')  # iterate through the test set
    error_count = 0.0
    mTest = len(test_files)
    for i in range(mTest):
        file_name = test_files[i].split('.')[0]
        class_number = file_name.split('_')[0]
        test_matrix = img2vector('data/testDigits'+'/'+test_files[i])
        classifierResult = knn(test_matrix, train_matrix, train_labels, 3)
        print("the classifier came back with: {}, the real answer is: {}" .format(classifierResult, class_number))
        if (classifierResult != class_number): error_count += 1.0
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count / float(mTest)))


if __name__ == '__main__':
    main()