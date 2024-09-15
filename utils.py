import os
import pandas as pd
import numpy as np
import pickle
import random


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_sentence_tag_train(lines, accident):
    sentences = []
    tags = []
    for i in range(len(lines)):
        if not np.isnan(lines.at[i, accident]):
            tags.append(lines.at[i, accident])
            sentences.append(lines.iloc[i, 0].strip().split())       
    all_tags = list(set(tags))

    indexList = random.sample(list(range(len(sentences))), len(sentences))
    n_of_tr = int(len(sentences)*0.9)
    n_of_samples = int(len(sentences)*0.0)
    sentences_tr, tags_tr = [sentences[i] for i in indexList[:n_of_tr]], [tags[i] for i in indexList[:n_of_tr]]
    sentences_val, tags_val = [sentences[i] for i in indexList[n_of_tr:]], [tags[i] for i in indexList[n_of_tr:]]
    print(len(sentences), len(tags))
    return sentences_val, tags_val, sentences, tags, all_tags


def create_sentence_test(lines, accident):
    sentences = []
    tags = []
    for i in range(len(lines)):
        if not np.isnan(lines.at[i, accident]):
            tags.append(lines.at[i, accident])
            sentences.append(lines.iloc[i, 0].strip().split())     
    print(len(sentences), len(tags))
    return sentences, tags


def parse_ourData_newformat(train_path, test_path, save_dir=None):

    # Choose a lading indicator:
    # {'DangerousEquipment','EquipmentProximity','WorkerMovement','Slips','LeadingEdge','Confinedspace','Hazardousmaterial',
    # 'Heavymaterialmovement','Electricalsproximity','Hightempertureoperation'}
    accident = 'Hightempertureoperation'

    lines_train = pd.read_csv(train_path)
    lines_test = pd.read_csv(test_path)

    Data = {}

    sentences_test, tags_test = create_sentence_test(lines_test,accident)
    sentences_val, tags_val, sentences_tr, tags_tr, all_tags = create_sentence_tag_train(lines_train,accident)
    
    Data["tr_inputs"], Data["tr_tags"] = sentences_tr, tags_tr
    Data["val_inputs"], Data["val_tags"] = sentences_val, tags_val
    Data["test_inputs"], Data["test_tags"] = sentences_test, tags_test
    
    Data["tr_tokens"] = Data["tr_inputs"]
    Data["val_tokens"] = Data["val_inputs"]
    Data["test_tokens"] = Data["test_inputs"]

    tag2 = {}
    tag_rev2 = {}

    for i, tag in enumerate(all_tags):
        tag_rev2[tag] = i + 1
        tag2[i + 1] = tag
    print("Labels: ", all_tags)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_dir is not None:
        save_obj(Data, save_dir + '/Data')
        save_obj(tag2, save_dir + '/tag2')
        save_obj(tag_rev2, save_dir + '/tag_rev2')


