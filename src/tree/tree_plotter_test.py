import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(nodeTxt, centerPt, parentPt, nodeType):
    create_plot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                             xytext=centerPt, textcoords='axes fraction',
                             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def get_leafs_num(mytree):
    num_leafs = 0
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]) == dict:
            num_leafs += get_leafs_num(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(mytree):
    max_depth = 0
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]) == dict:
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth: max_depth = this_depth
    return max_depth


def plot_mid_text(cntr_pt, parent_pt, txt_str):
    xmid = (parent_pt[0] - cntr_pt[0]) / 2 + cntr_pt[0]
    ymid = (parent_pt[1] - cntr_pt[1]) / 2 + cntr_pt[1]
    create_plot.ax1.text(xmid, ymid, txt_str)


def plot_tree(mytree, parent_pt, node_text):
    num_leafs = get_leafs_num(mytree)
    depth = get_tree_depth(mytree)
    first_str = list(mytree.keys())[0]
    cntr_pt = plot_tree.xoff + (1 + float(num_leafs)) / 2 / plot_tree.totalW, plot_tree.yoff
    plot_mid_text(cntr_pt, parent_pt, node_text)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = mytree[first_str]
    plot_tree.yoff = plot_tree.yoff - 1 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]) == dict:
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xoff = plot_tree.xoff + 1 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xoff, plot_tree.yoff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), cntr_pt, str(key))
    plot_tree.yoff = plot_tree.yoff + 1.0 / plot_tree.totalD


def create_plot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_leafs_num(intree))
    plot_tree.totalD = float(get_tree_depth(intree))
    plot_tree.xoff = -0.5/plot_tree.totalW
    plot_tree.yoff = 1.0
    plot_tree(intree, (0.5,1.0), '')
    plt.show()


def retrieve_tree(i):
    trees_list = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return trees_list[i]

def classify(mytree,feat_labels, test_vec):
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key])==dict:
                class_label = classify(second_dict[key],feat_labels,test_vec)
            else: class_label = second_dict[key]
    return class_label

if __name__ == '__main__':
    # mytree = retrieve_tree(1)
    # create_plot(mytree)
    tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    features = [1, 0, 1]
    labels = ['head', 'flippers', 'no surfacing']
    print(classify(tree,labels,features))
