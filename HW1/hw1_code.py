from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import math


def load_data():
    real_df = pd.read_csv('data/clean_real.txt', header=None, index_col=False)
    fake_df = pd.read_csv('data/clean_fake.txt', header=None, index_col=False)
    real_df['label'] = 1
    fake_df['label'] = 0
    frames = [real_df, fake_df]
    combine_df = pd.concat(frames)
    y = combine_df['label']
    combine_df.drop('label', axis=1, inplace=True)
    vc = CountVectorizer()
    all = vc.fit_transform(combine_df[0])
    x_train, x_temp, y_train, y_temp = train_test_split(all, y, test_size=0.3,
                                                        random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                    test_size=0.5,
                                                    random_state=55)
    return x_train, y_train, x_val, y_val, x_test, y_test


def select_model(x_train, y_train, x_val, y_val):
    max_depth = [3, 5, 7, 9, 11]
    criteria = ['gini', 'entropy']
    accuracy = []
    for i in range(len(max_depth)):
        for j in range(len(criteria)):
            dt = DecisionTreeClassifier(criterion=criteria[j],
                                        max_depth=max_depth[i])
            dt = dt.fit(x_train, y_train)
            pre_val = dt.predict(x_val)
            acc = metrics.accuracy_score(y_val, pre_val)
            accuracy.append(acc)
            # print("criteria: ", criteria[j], "  max_depth: ", max_depth[i], "  accuracy: ",  metrics.accuracy_score(y_val, pre_val))

        ##---------------------enable below part to plot the tree-----------------------------------------
            # if (criteria[j] == 'gini' and max_depth[i] == 11):
            #     print(tree.export_text(dt))



def compute_information_gain(key_index, x, y):
    all_real = len(y[y[0] == 1])
    all_fake = len(y[y[0] == 0])
    total = all_real + all_fake
    right_real = 0
    right_fake = 0
    left_real = 0
    left_fake = 0
    s = x[key_index]
    for i in range(len(s)):
        if (s[i] == 0 & y[i] == 0):
            left_fake += 1
        if (s[i] == 0 & y[i] == 1):
            left_real += 1
        if (s[i] == 1 & y[i] == 0):
            right_fake += 1
        if (s[i] == 1 & y[i] == 1):
            right_real += 1
    right_total = right_fake + right_real
    left_total = left_fake + left_real
    before = -(all_real / total) * math.log2((all_real / total)) - (
                all_fake / total) * math.log2((all_fake / total))
    after = (-(left_real / left_total) * math.log2((left_real / left_total)) - (
                left_fake / left_total) * math.log2(
        (left_fake / left_total))) * (left_total / total) + (-(right_real / right_total) * math.log2((right_real / right_total)) - (
            right_fake / right_total) * math.log2(
        (right_fake / right_total))) * (right_total / total)

    return (before - after)


if __name__ == '__main__':

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    # print(x_val.toarray())
    select_model(x_train, y_train, x_val, y_val)
