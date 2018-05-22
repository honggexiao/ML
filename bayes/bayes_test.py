import numpy as np
import bayes2

if __name__ == '__main__':
    list_posts, list_classes = bayes2.load_data_set()
    vocab_list = bayes2.create_vocab_list(list_posts)
    train_mat = []
    for postdoc in list_posts:
        train_mat.append(bayes2.create_words_vec(vocab_list, postdoc))
    p_vec_0, p_vec_1, p_ab = bayes2.train_bayes(np.array(train_mat), np.array(list_classes))
    test_entry=['love','my','dalmation']
    print(bayes2.create_words_vec(vocab_list, test_entry))
    this_doc = np.array(bayes2.create_words_vec(vocab_list, test_entry))
    print(this_doc)
    print(test_entry, 'classified as:', bayes2.classify_bayes(this_doc, p_vec_0, p_vec_1, p_ab))


