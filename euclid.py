import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
# [[plt.scatter(ii[0],ii[1], s=100, color = i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()
def knn(data,predict,k = 3):
    if len(data) >= k:
        warnings.warn('idiot!')
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt((features[0] - predict[0])**2 + (features[1] - predict[1])**2)
            euclid_dist = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclid_dist, group])

    votes = [i[1] for i in sorted(distances)[:k]]
   # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

accuracies = []

for i in range(5):

    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?',-99999, inplace=True)
    df.drop(['id'],1, inplace=True)
    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size* len(full_data))]
    test_data = full_data[-int(test_size* len(full_data)):]

    #now we want to populate these dictionaries

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    #now we need to pass the info to kNN

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = knn(train_set, data, k=5)
            if group == vote:
                correct +=1
            #else:
                #print(confidence)
            total +=1

   # print('Accuracy:', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
#now we want to compare this to scikit-learn







