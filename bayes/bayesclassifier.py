import numpy as np

class BayesClassifier:
    """this is bayes classifer, given feature vector,
     output is class labels"""
    def __init__(self, feature_num, labels_list, feature_val_list):
        self.features_num = feature_num
        self.labels_list = labels_list
        self.feature_val_list = list(feature_val_list)

    def process_data(self, data_file):
        class_count = {}
        cond_dic = {}
        m = len(self.labels_list)
        for i in range(m):
            label = self.labels_list[i]
            cond_dic[label] = {}
            for j, feature_vals in enumerate(self.feature_val_list):
                cond_dic[label][j] = {}
                for val in feature_vals:
                    cond_dic[label][j][val] = 0
        with open(data_file) as f:
            samples = f.readlines()
            self.samples_num = len(samples)
            for line in samples:
                sample = line[:-1].split(',')
                label = sample[-1]
                class_count[label] = class_count.get(label, 0) + 1
                for j, val in enumerate(sample[0:-1]):
                    cond_dic[label][j][val] = cond_dic[label][j][val] +1
        self.class_count = class_count
        self.cond_dic = cond_dic

    def classify(self, sample):
        argm_prob = 0
        sample_label = self.labels_list[0]
        for label in self.labels_list:
            label_prob = self.class_count[label]/float(self.samples_num)
            cond_prob = 1.0
            for j, val in enumerate(sample):
                cond_prob *= self.cond_dic[label][j][val]/1.0
            prob = label_prob * cond_prob
            if prob > argm_prob:
                argm_prob = prob
                sample_label = label
        return sample_label, argm_prob


if __name__ == '__main__':
    datafile = 'data.txt'
    classifier = BayesClassifier(2, [1, -1], [[1, 2, 3], ['S', 'M', 'L']])
    classifier.process_data(datafile)
    sample = [2, 'S']
    label, prob = classifier.classify(sample)
