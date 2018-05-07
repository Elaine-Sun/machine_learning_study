import numpy as np
import matplotlib.pyplot as plt
import operator


def read_dataset(fp):
    fr = open(fp)
    lines = fr.readlines()
    m = len(lines)
    feature_set = np.zeros((m, 3))
    level_set = []
    index = 0
    for line in lines:
        line = line.strip().split('\t')
        feature_set[index, :] = line[0:3]
        level_set.append(int(line[3]))
        index += 1
    return feature_set, level_set


def get_visualization(fp):
    feature_set, level_set = read_dataset(fp=fp)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(feature_set[:, 0], feature_set[:, 1], 15.0 * np.array(level_set), 15.0 * np.array(level_set))
    plt.show()


def normalize_data(fp):
    feature_set, level_set = read_dataset(fp=fp)
    m = feature_set.shape[0]
    data_min = feature_set.min(0)
    data_max = feature_set.max(0)
    range = data_max - data_min
    norm_data = (feature_set - np.tile(data_min, (m, 1))) / range
    return norm_data, level_set


def knn(test_data, train_data, level_set, k):
    distances = (((np.tile(test_data, (train_data.shape[0], 1)) - train_data) ** 2).sum(axis=1)) ** 0.5
    sorted_distance = distances.argsort()
    class_count = {}
    for i in range(k):
        level = level_set[sorted_distance[i]]
        class_count[level] = class_count.get(level, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def main():
    test_rate = 0.5
    fp = 'C:/Users/sunyi/Desktop/work/study/ai/Ch02/datingTestSet2.txt'
    feature_set, level_set = normalize_data(fp=fp)
    n = feature_set.shape[0]
    m = int(test_rate * n)
    err_count = 0
    for i in range(m):
        test_data = feature_set[i, :]
        train_data = feature_set[m:n, :]
        result = knn(test_data=test_data, train_data=train_data, level_set=level_set[m:n], k=3)
        print('识别出的分类为：{}，实际结果为：{}'.format(result, level_set[i]))
        if result != level_set[i]: err_count += 1
    print('错误数：{}'.format(err_count))
    print('错误率：{}%'.format(round(err_count / (n - m) * 100,2)))


if __name__ == '__main__':
    main()
