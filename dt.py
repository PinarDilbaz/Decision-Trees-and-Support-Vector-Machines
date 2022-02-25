import numpy as np
import math 
def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    number_of_bucket = len(bucket)
    summ = sum(bucket)
    entropy = 0
    for i in range(number_of_bucket):
        if (bucket[i] != 0):
            x = bucket[i] / summ
            y = math.log2(bucket[i] / summ)
            entropy = entropy + (-x*y)
    return entropy


def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    summ_parent = sum(parent_bucket)
    summ_left = sum(left_bucket)
    summ_right = sum(right_bucket)

    entropy_parent = entropy(parent_bucket)
    entropy_left = entropy(left_bucket)
    entropy_right = entropy(right_bucket)

    child_left = summ_left/summ_parent*entropy_left
    child_right = summ_right/summ_parent*entropy_right

    child = child_left + child_right
    information_gain = entropy_parent - child

    return information_gain


def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    summ = sum(bucket)
    gini_index = 0

    if (summ != 0):
        for b in bucket:
            gini_index = gini_index +(b / summ) ** 2
    gini_index = 1 - gini_index

    return gini_index


def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    summ_left = sum(left_bucket)
    summ_right = sum(right_bucket)
    total = summ_right + summ_left

    gini_left = gini(left_bucket)
    gini_right = gini(right_bucket)

    child_left = summ_left/total*gini_left
    child_right = summ_right/total*gini_right
    average_gini = child_left + child_right

    return average_gini

def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    arr = []
    new_arr = []
    sorted_label = []
    l = []
    r = []
    heuristic_arr = []
    for i in range(len(data)):
        arr.append(data[i][attr_index])
    sorted_arr = sorted(arr)

    for i in range(len(sorted_arr)):
        index = arr.index(sorted_arr[i])
        sorted_label.append(labels[index])
    p = np.asarray(sorted_label)

    for i in range(len(sorted_arr)-1):
        new_value = (sorted_arr[i]+sorted_arr[i+1])/2
        new_arr.append(new_value)

    for i in range(len(new_arr)):
        l = []
        r = []
        parent_bucket = [0] * num_classes
        left_bucket = [0] * num_classes
        right_bucket = [0] * num_classes
    
        for j in range(len(sorted_arr)):
            if(sorted_arr[j]<new_arr[i]):
                l.append(sorted_label[j])
            elif(sorted_arr[j]>=new_arr[i]):
                r.append(sorted_label[j])
        l = np.asarray(l)
        r = np.asarray(r)
        for j in range(num_classes):
            total = 0
            for val in l:
                if (j == val):
                    total = total + 1
            left_bucket[j] = total
            total = 0
            for val in r:
                if (j == val):
                    total = total + 1
            right_bucket[j] = total
            total = 0
            for val in p:
                if (j == val):
                    total = total + 1
            parent_bucket[j] = total
        if (heuristic_name == 'info_gain'):
            information_gain = info_gain(parent_bucket, left_bucket, right_bucket)
            heuristic_arr.append([new_arr[i], information_gain])

        elif (heuristic_name == 'avg_gini_index'):
            average_gini = avg_gini_index(left_bucket, right_bucket)
            heuristic_arr.append([new_arr[i], average_gini])
    heuristic_arr = np.asarray(heuristic_arr)

    return heuristic_arr

def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    chi_squared_value = 0
    summ_left = sum(left_bucket)
    summ_right = sum(right_bucket)
    total = summ_right + summ_left
    number_of_bucket = len(right_bucket)
    degree_of_freedom = number_of_bucket
    for i in range(number_of_bucket):
        if (left_bucket[i] != 0 or right_bucket[i] != 0):
            value = left_bucket[i] + right_bucket[i]
            l = summ_left/total
            r = summ_right/total
            x = (value*l)
            y = (value*r)
            left = ((left_bucket[i]-x)**2)/x
            right = ((right_bucket[i]-y)**2)/y
            chi_squared_value = chi_squared_value + left + right
            
        elif(left_bucket[i] == 0 and right_bucket[i] == 0):
            degree_of_freedom = degree_of_freedom - 1
    degree_of_freedom = degree_of_freedom - 1

    return chi_squared_value,degree_of_freedom

if __name__ == "__main__":
    train_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/dt/train_set.npy')
    train_lbs = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/dt/train_labels.npy') 
    test_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/dt/test_set.npy')
    test_labels = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/dt/test_labels.npy') 