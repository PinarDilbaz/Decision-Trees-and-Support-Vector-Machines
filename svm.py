import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def svm_task1(train_set,train_labels):
    c_values = [0.01, 0.1, 1, 10, 100]
    for c in c_values:
        clf = SVC(kernel='linear', C=c)
        clf = clf.fit(train_set, train_labels)
        draw_svm(clf, train_set, train_labels, train_set[:, 0].min(), train_set[:, 0].max(), train_set[:, 1].min(), train_set[:, 1].max())

def svm_task2(train_set,train_labels):
    kernels = ['linear','rbf', 'poly', 'sigmoid']
    for k in kernels:
        clf = SVC(kernel=k, C=1)
        clf = clf.fit(train_set, train_labels)
        draw_svm(clf, train_set, train_labels, train_set[:, 0].min(), train_set[:, 0].max(), train_set[:, 1].min(), train_set[:, 1].max())

def svm_task3(train_set,train_lbs,test_set,test_labels):
    #for train set
    train_set = train_set / 256.
    count, x_axis, y_axis = train_set.shape
    train_set = train_set.reshape((count, x_axis * y_axis))
    #for test set
    test_set = test_set / 256.
    count, x_axis, y_axis = test_set.shape
    test_set = test_set.reshape((count, x_axis * y_axis))

    kernels = ['linear','rbf', 'poly', 'sigmoid']
    c_values = [0.01, 0.1, 1, 10, 100]
    gamma_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    best_param = []
    best_param.append(['linear',0.01,0.00001])
    temp = 0.0
    
    for k in kernels:
        for c in c_values:
            for g in gamma_values:
                clf = SVC(C=c, kernel=k, gamma=g)
                clf.fit(train_set, train_lbs)
                accuracy = clf.score(train_set, train_lbs)
                print("Kernel => %s , C => %f , Gamma => %f and Validation Accuracy is %f"% (k, c, g, accuracy))
                if accuracy > temp:
                    best_param.pop(0)
                    best_param.append([k,c,g])
                    temp = accuracy
    clf2 = SVC(C=best_param[0][1], kernel=best_param[0][0], gamma=best_param[0][2])
    clf2.fit(train_set, train_lbs)
    accuracy = clf2.score(test_set, test_labels)    
    print(best_param)
    print("Kernel => %s , C => %f , Gamma => %f and Test Accuracy is %f"% (best_param[0][0], best_param[0][1], best_param[0][2], accuracy))        
    
def svm_task4(train_set,train_lbs,test_set,test_labels):
    print(" Task-4")
    print(" 1-) Imbalanced train data")
    print(" 2-) Oversample the minority class")
    print(" 3-) Undersample the majority class")
    print(" 4-) With class_weight")
    selection = int(input("Please enter your choice: "))
    if(selection == 1):
        #for train set
        train_set = train_set / 256.
        count, x_axis, y_axis = train_set.shape
        train_set = train_set.reshape((count, x_axis * y_axis))
        #for test set
        test_set = test_set / 256.
        count, x_axis, y_axis = test_set.shape
        test_set = test_set.reshape((count, x_axis * y_axis))

        k = 'rbf'
        c = 1
        clf = SVC(C=c, kernel=k)
        clf.fit(train_set, train_lbs)
        accuracy = clf.score(test_set, test_labels)

        p = clf.predict(test_set)
        confusionMatrix = confusion_matrix(test_labels, p)
        print ("Test Accuracy is %f "% (accuracy))
        print("Confusion Matrix")
        print(confusionMatrix)
    elif(selection == 2):
        oversampled_train_set = []
        oversampled_train_lbs = []
        lbs = []
        arr = []
        number_of_train_set = len(train_set)
        for i in range(number_of_train_set):
            oversampled_train_set.append(train_set[i])
            oversampled_train_lbs.append(train_lbs[i])
        for i in range(number_of_train_set):
            if oversampled_train_lbs[i] not in lbs:
                lbs.append(oversampled_train_lbs[i])
        for i in lbs:
            summ = 0
            for j in train_lbs:
                if (i==j):
                    summ = summ + 1
            arr.append(summ)
        for i in range(number_of_train_set):
            if (oversampled_train_lbs[i] == 0):
                for j in range(int(arr[0]/arr[1])):
                    oversampled_train_set.append(oversampled_train_set[i])
                    oversampled_train_lbs.append(oversampled_train_lbs[i])
        oversampled_train_set = np.asarray(oversampled_train_set)
        oversampled_train_lbs  = np.asarray(oversampled_train_lbs)
        #for train set
        count, x_axis, y_axis = oversampled_train_set.shape
        oversampled_train_set = oversampled_train_set.reshape((count, x_axis * y_axis))
        #for test set
        count, x_axis, y_axis = test_set.shape
        test_set = test_set.reshape((count, x_axis * y_axis))

        k = 'rbf'
        c = 1
        clf = SVC(C=c, kernel=k)
        clf.fit(oversampled_train_set, oversampled_train_lbs)
        accuracy = clf.score(test_set, test_labels)

        p = clf.predict(test_set)
        confusionMatrix = confusion_matrix(test_labels, p)
        print ("Test Accuracy is %f "% (accuracy))
        print("Confusion Matrix")
        print(confusionMatrix)

    elif(selection == 3):
        undersampled_train_set = []
        undersampled_train_lbs = []
        lbs = []
        arr = []
        number_of_train_set = len(train_set)
        for i in range(number_of_train_set):
            undersampled_train_set.append(train_set[i])
            undersampled_train_lbs.append(train_lbs[i])
        for i in range(number_of_train_set):
            if undersampled_train_lbs[i] not in lbs:
                lbs.append(undersampled_train_lbs[i])
        for i in lbs:
            summ = 0
            for j in train_lbs:
                if (i==j):
                    summ = summ + 1
            arr.append(summ)
        i = 0
        while (arr[1] != arr[0]):
            if undersampled_train_lbs[i] == 1:
                arr[0] = arr[0] - 1
                undersampled_train_set.pop(i)
                undersampled_train_lbs.pop(i)
            elif undersampled_train_lbs[i] != 1:
                i = i + 1

        undersampled_train_set = np.asarray(undersampled_train_set)
        undersampled_train_lbs  = np.asarray(undersampled_train_lbs)
        #for train set
        count, x_axis, y_axis = undersampled_train_set.shape
        undersampled_train_set = undersampled_train_set.reshape((count, x_axis * y_axis))
        #for test set
        count, x_axis, y_axis = test_set.shape
        test_set = test_set.reshape((count, x_axis * y_axis))

        k = 'rbf'
        c = 1
        clf = SVC(C=c, kernel=k)
        clf.fit(undersampled_train_set, undersampled_train_lbs)
        accuracy = clf.score(test_set, test_labels)

        p = clf.predict(test_set)
        confusionMatrix = confusion_matrix(test_labels, p)
        print ("Test Accuracy is %f "% (accuracy))
        print("Confusion Matrix")
        print(confusionMatrix)
    elif(selection == 4):
        #for train set
        train_set = train_set / 256.
        count, x_axis, y_axis = train_set.shape
        train_set = train_set.reshape((count, x_axis * y_axis))
        #for test set
        test_set = test_set / 256.
        count, x_axis, y_axis = test_set.shape
        test_set = test_set.reshape((count, x_axis * y_axis))

        k = 'rbf'
        c = 1
        clf = SVC(C=c, kernel=k, class_weight='balanced')
        clf.fit(train_set, train_lbs)
        accuracy = clf.score(test_set, test_labels)

        p = clf.predict(test_set)
        confusionMatrix = confusion_matrix(test_labels, p)
        print ("Test Accuracy is %f "% (accuracy))
        print("Confusion Matrix")
        print(confusionMatrix)

def draw_svm(clf, x, y, x1_min, x1_max, x2_min, x2_max, target=None):
    """
    Draws the decision boundary of an svm.
    :param clf: sklearn.svm.SVC classifier
    :param x: data Nx2
    :param y: label N
    :param x1_min: minimum value of the x-axis of the plot
    :param x1_max: maximum value of the x-axis of the plot
    :param x2_min: minimum value of the y-axis of the plot
    :param x2_max: maximum value of the y-axis of the plot
    :param target: if target is set to path, the plot is saved to that path
    :return: None
    """
    y = y.astype(bool)
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 500),
                         np.linspace(x2_min, x2_max, 500))
    z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    disc_z = z > 0
    plt.clf()
    plt.imshow(disc_z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.RdBu, alpha=.3)
    plt.contour(xx, yy, z, levels=[-1, 1], linewidths=2,
                linestyles='dashed', colors=['red', 'blue'], alpha=0.5)
    plt.contour(xx, yy, z, levels=[0], linewidths=2,
                linestyles='solid', colors='black', alpha=0.5)
    positives = x[y == 1]
    negatives = x[y == 0]
    plt.scatter(positives[:, 0], positives[:, 1], s=50, marker='o', color="none", edgecolor="black")
    plt.scatter(negatives[:, 0], negatives[:, 1], s=50, marker='s', color="none", edgecolor="black")
    sv_label = y[clf.support_]
    positive_sv = x[clf.support_][sv_label]
    negative_sv = x[clf.support_][~sv_label]
    plt.scatter(positive_sv[:, 0], positive_sv[:, 1], s=50, marker='o', color="white", edgecolor="black")
    plt.scatter(negative_sv[:, 0], negative_sv[:, 1], s=50, marker='s', color="white", edgecolor="black")
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.gca().set_aspect('equal', adjustable='box')
    if target is None:
        plt.show()
    else:
        plt.savefig(target)

if __name__ == "__main__":
    print(" Support Vector Machines")
    print(" 1-) Task1")
    print(" 2-) Task2")
    print(" 3-) Task3")
    print(" 4-) Task4")
    selection = int(input("Please enter your choice: "))
    if(selection == 1):
        #for task1 data load
        train_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task1/train_set.npy')
        train_lbs = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task1/train_labels.npy') 
        svm_task1(train_set,train_lbs)
    elif(selection == 2):
        #for task2 data load
        train_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task2/train_set.npy')
        train_lbs = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task2/train_labels.npy') 
        svm_task2(train_set,train_lbs)
    elif(selection == 3):
        #for task3 data load
        train_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task3/train_set.npy')
        train_lbs = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task3/train_labels.npy') 
        test_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task3/test_set.npy')
        test_labels = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task3/test_labels.npy') 
        svm_task3(train_set,train_lbs,test_set,test_labels)
    elif(selection == 4):
        #for task4 data load
        train_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task4/train_set.npy')
        train_lbs = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task4/train_labels.npy') 
        test_set = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task4/test_set.npy')
        test_labels = np.load('C:/Users/ASUS/Desktop/Cng409/hw3/svm/task4/test_labels.npy') 
        svm_task4(train_set,train_lbs,test_set,test_labels)
        

        