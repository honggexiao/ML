class BayesClassifier:
    """this is bayes classifer, given feature vector,
     output is class labels"""
    def __init__(self, labels_list, feature_val_list):
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
                label = int(sample[-1])
                class_count[label] = class_count.get(label, 0) + 1
                for j, val in enumerate(sample[0:-1]):
                    try:
                        val_tmp = int(val)
                    except Exception as e:
                        val_tmp = val
                    cond_dic[label][j][val_tmp] = cond_dic[label][j][val_tmp] + 1
        self.class_count = class_count
        self.cond_dic = cond_dic

    def classify(self, sample, k_lambda=0):
        argm_prob = 0
        sample_label = self.labels_list[0]
        for label in self.labels_list:
            label_cnt = self.class_count[label]
            label_prob = (label_cnt + k_lambda)/float(self.samples_num + len(self.labels_list)*k_lambda)
            cond_prob = 1.0
            for j, val in enumerate(sample):
                cond_prob *= (float(self.cond_dic[label][j][val] + k_lambda)/(label_cnt + k_lambda*len(self.feature_val_list[j])))
            prob = label_prob * cond_prob
            if prob > argm_prob:
                argm_prob = prob
                sample_label = label
        return sample_label, argm_prob


if __name__ == '__main__':
    datafile = 'data.txt'
    classifier = BayesClassifier([1, -1], [[1, 2, 3], ['S', 'M', 'L']])
    classifier.process_data(datafile)
    sample = [2, 'S']
    label, prob = classifier.classify(sample, 1)
    print('label = %d' %label)
    print('prob= %f' %prob)
