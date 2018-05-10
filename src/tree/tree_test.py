from math import log
import operator


def get_entropy(dataset):
    sample_dict = {}
    total_len = len(dataset)
    for sample in dataset:
        if sample[-1] not in sample_dict.keys():
            sample_dict[sample[-1]] = 0
        sample_dict[sample[-1]] += 1
    entropy = 0
    for label in sample_dict.keys():
        label_rate = sample_dict[label] / total_len
        entropy -= label_rate * log(label_rate, 2)
    return entropy


def split_dataset(dataset, loc, value):
    dataset_list = []
    for sample in dataset:
        if sample[loc] == value:
            sample = sample[:loc] + sample[loc + 1:]
            dataset_list.append(sample)
    return dataset_list


def choose_best_feature_to_split(dataset):
    feature_num = len(dataset[0]) - 1
    origin_entropy = get_entropy(dataset)
    max_entropy = 0
    best_feature = -1
    for i in range(feature_num):
        feature_list = [data[i] for data in dataset]
        feature_set = set(feature_list)
        entropy = 0
        for feature in feature_set:
            subdataset = split_dataset(dataset, i, feature)
            label_rate = len(subdataset) / len(dataset)
            entropy += label_rate * get_entropy(subdataset)
        process_entropy = origin_entropy - entropy
        if max_entropy < process_entropy:
            max_entropy = process_entropy
            best_feature = i
    return best_feature


def majority_count(class_list):
    class_count = {}
    for clas in class_list:
        if clas not in class_count:
            class_count[clas] = 0
        class_count[clas] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count


def create_tree(dataset, labels):
    class_list = [sample[-1] for sample in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    best_feature_loc = choose_best_feature_to_split(dataset=dataset)
    best_feature_label = labels[best_feature_loc]
    mytree = {best_feature_label: {}}
    del (labels[best_feature_loc])
    feature_values = set([sample[best_feature_loc] for sample in dataset])
    for value in feature_values:
        sublabels = labels[:]
        mytree[best_feature_label][value] = create_tree(
            dataset=split_dataset(dataset=dataset, loc=best_feature_loc, value=value), labels=sublabels)
    return mytree


if __name__ == '__main__':
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # print(split_dataset(dataset=dataset,loc=0,value=1))
    # print(get_entropy(dataset=dataset))
    # print(choose_best_feature_to_split(dataset=dataset))
    print(create_tree(dataset=dataset, labels=labels))
