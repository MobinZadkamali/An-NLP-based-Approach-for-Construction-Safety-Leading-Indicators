import torch
import os, sys
from models.transformer.seqTagger import tokenize_and_pad_text_for_train
from transformers import AutoConfig
from models.transformer import tagger_trainer
from models.transformer.tagger_trainer import validation
from utils import load_obj
import project_statics
from transformers import AutoTokenizer
from models.transformer.BERT_sentence_classification import Bert
from prettytable import PrettyTable

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))




def get_Data(Data):
    Data["tr_inputs"] = torch.tensor(Data["tr_inputs"])
    Data["val_inputs"] = torch.tensor(Data["val_inputs"])
    Data["test_inputs"] = torch.tensor(Data["test_inputs"])

    Data["tr_tags"] = torch.tensor(Data["tr_tags"])
    Data["val_tags"] = torch.tensor(Data["val_tags"])
    Data["test_tags"] = torch.tensor(Data["test_tags"])

    Data["tr_masks"] = torch.tensor(Data["tr_masks"])
    Data["val_masks"] = torch.tensor(Data["val_masks"])
    Data["test_masks"] = torch.tensor(Data["test_masks"])
    return Data

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
        

if __name__ == '__main__':

    # set to  CUDA if you have GPU o.w. cpu
    # device = torch.device("cpu")
    device = torch.device("cuda")
    save_path = 'test_bert'
    """
    sentences is a list of sentences each sentence list of tokens: [['جمله','دوم','تست'],['جمله','اول','تست']]
    tags is list of tags: [['O','O','O'],['O','O','O']]
    dict2 is dictionary of id(1 to N) to tag_text
    ditc2_rev is dictionary of tag_text to id (1 to N) 
    """
    data_path = project_statics.Leading_indicator_pickle_files
    Data = load_obj(data_path + '/Data')
    tag2 = load_obj(data_path + '/tag2')
    tag_rev2 = load_obj(data_path + '/tag_rev2')
    print('There are ', len(Data["tr_inputs"]), 'sentences for training.')


    # find best iteration number for your self
    epochs = 20
    # batch size
    bs = 128
    # learning rate
    lr = 5e-5

    xlmr_model = 'xlm-roberta-base'
    parsbert = "HooshvareLab/bert-base-parsbert-uncased"
    mbert = 'bert-base-multilingual-cased'

    ## choose the pretrained language model
    model_name = parsbert

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ## max_len must be < 510, if too small there would be warnings about truncation and there will be truncation
    max_len = 100
    Data["tr_inputs"], Data["tr_lens"], tokenized_sentences, Data["tr_tags"], Data["tr_starts"] = tokenize_and_pad_text_for_train(Data["tr_inputs"], Data["tr_tags"], tokenizer, max_len=max_len,tag_rev2=tag_rev2)
    
    Data["val_inputs"], Data["val_lens"], tokenized_sentences, Data["val_tags"], Data["val_starts"] = tokenize_and_pad_text_for_train(Data["val_inputs"], Data["val_tags"], tokenizer,
                                                        max_len=max_len,tag_rev2=tag_rev2)
    Data["test_inputs"], Data["test_lens"], tokenized_sentences, Data["test_tags"], Data["test_starts"] = tokenize_and_pad_text_for_train(Data["test_inputs"], Data["test_tags"], tokenizer,
                                                         max_len=max_len,tag_rev2=tag_rev2)
    print('tokenized')

    # set this to true if you want to update the whole model
    FULL_FINETUNING = False

    config = AutoConfig.from_pretrained(model_name)
    config.model_name = model_name
    config.classifier_dropout = 0.1
    config.use_gpu = True
    config.output_attentions=True
    config.max_len = max_len

    """
    choose your model class: 
    """

    model_class = Bert

    # print(config)
    Data["tr_masks"] = [[i < Data["tr_lens"][j] + 2 for i in range(len(ii))] for j, ii in enumerate(Data["tr_inputs"])]
    Data["tr_inputs"] = Data["tr_inputs"].astype('int64')

    Data["val_masks"] = [[i < Data["val_lens"][j] + 2 for i in range(len(ii))] for j, ii in
                         enumerate(Data["val_inputs"])]
    Data["val_inputs"] = Data["val_inputs"].astype('int64')

    Data["test_masks"] = [[i < Data["test_lens"][j] + 2 for i in range(len(ii))] for j, ii in
                          enumerate(Data["test_inputs"])]
    Data["test_inputs"] = Data["test_inputs"].astype('int64')


    Data = get_Data(Data)

    config.tag2 = tag2

    model = model_class.from_pretrained(model_name, config=config,
                                        tag_label_lst=tag2.keys())

    count_parameters(model)

    """
    for precision recall NER_validation otherwise use POS_validation for  accuracy
    """

    val_fun = validation

    tagger_trainer.run_transformer_trainer(Data, batch_size=bs, FULL_FINETUNING=FULL_FINETUNING, model=model,
                                           tokenizer=tokenizer,
                                           device=device, validation_func=val_fun,
                                           lr=lr, epochs=epochs, save_dir=save_path)