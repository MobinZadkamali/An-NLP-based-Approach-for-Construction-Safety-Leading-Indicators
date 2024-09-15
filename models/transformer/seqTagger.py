from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def get_ouputs(model, dataloader: DataLoader, device: torch.device):
    """

    computes for evaluation phase in training

    :param PreTrainedModel model: Bert\RoBERTa for token classification
    :param dataloader: dataloader containing the data
    :param device: torch device
    :return:
    """

    # Put the model into evaluation mode
    model.eval()

    predictions, true_labels = [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_tags, lens = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, lens=lens, device=device)
        # Move logits and labels to CPU
        logits = outputs[1][0]
  
        if type(logits) == list:
            logits = np.array(logits)
        else:
            logits = logits.detach().cpu().numpy()

        ids = b_tags.to('cpu').numpy()

        predictions.extend(np.argmax(logits, axis=1))
        true_labels.extend(ids)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    return predictions, true_labels


def tag_and_tokens_to_original_form(label_indices, dict2, starts, true_labels):
    """

    :param label_indices: labels indices to match to new tokens
    :param dict2: a dictionary mapping from indices to labels
    :param starts: starts[i]=1 is position i is start of a new word o.w. 0
    :param true_labels: true label indices
    :return: new_labels, new_true
    """
    new_tokens, new_labels, new_true = [], [], []
    for label_idx, start, true_id in zip(label_indices, starts, true_labels):
        if start == 0:
            pass
        else:
            new_labels.append(dict2[label_idx + 1])

            new_true.append(dict2[true_id + 1])

    return new_labels, new_true


def tokenize_and_pad_text(sentences, tokenizer, need_tokenization, max_len):
    """
    tokenizes and pads the sentence

    :param sentences: list of string, input sentences
    :param tokenizer: Bert|RoBERTa tokenizer
    :param max_len: the outputs will have length maxlen, but sentences with size bigger than maxlen - 2  will be truncated (two for [CLS] and [SEP])
    :return: tokenized_sentences_ids ,lens, tokenized_sentences,original_sentences, starts


    """
    tokenized_sentences = []
    tokenized_sentences_ids = []
    lens = []
    original_sentences = []
    starts = []
    for i in range(len(sentences)):

        tokenized_sentence = []
        orig_sen = []
        if need_tokenization==True:
            sentence = sentences[i].split(' ')
        else:
            sentence = sentences[i]
        start = []

        for word in sentence:
            tokenized_word = tokenizer.tokenize(word)
            if len(tokenized_word) == 1:
                start.append(1)
                if tokenized_word[0] != '[UNK]':
                    orig_sen.append(tokenized_word[0])
                else:
                    orig_sen.append(word)
            elif len(tokenized_word) > 0:
                start.append(1)
                for k in range(len(tokenized_word) - 1):
                    start.append(0)
                if '[UNK]' in tokenized_word:
                    orig_sen.extend([word] * len(tokenized_word))
                else:
                    orig_sen.extend(tokenized_word)
            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
        original_sentences.append(orig_sen)
        starts.append(start)
        if len(tokenized_sentence) > max_len - 2:
            print('Warning : Size', len(tokenized_sentence), ' is bigger than maxlen - 2 , truncating index', i)
            tokenized_sentence = tokenized_sentence[:max_len - 2]

        lens.append(len(tokenized_sentence))
        # print('len is ', lens[-1])
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        tokenized_sentence.extend(["[PAD]"] * (max_len - len(tokenized_sentence)))
        tokenized_sentence_id = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokenized_sentences.append(tokenized_sentence)
        tokenized_sentences_ids.append(tokenized_sentence_id)
    return np.array(tokenized_sentences_ids, dtype='long'), np.array(lens,
                                        dtype='long'), tokenized_sentences, original_sentences, starts


def tokenize_and_pad_text_for_train(sentences, tags, tokenizer, max_len, tag_rev2):
    """

    tokenizes and pads the sentence to feed to training procedure

    :param sentences: list of string, input sentences
    :param tags: corresponding input tags, will be stretched if word are breaked  by tokenizer
    :param tokenizer: Bert|RoBERTa tokenizer
    :param max_len: the outputs will have length maxlen, but sentences with size bigger than maxlen - 2  will be truncated (two for [CLS] and [SEP])
    :return: tokenized_sentences_ids ,lens, tokenized_sentences,original_sentences, starts
    :param dict_rev2:a dictionary mapping from textual labels to indices
    :return: input_ids, lens, tokenized_sentences, tags_ids, tag_ids, starts
    """
    tokenized_sentences = []
    tokenized_sentences_ids = []
    tag_ids = []
    lens = []
    starts = []
    for i in range(len(sentences)):

        tokenized_sentence = []
        start = []
        sentence = sentences[i]
        for word in sentence:
            # print(word)
            tokenized_word = tokenizer.tokenize(word)

            # Add the tokenized word to the final tokenized word list
            if len(tokenized_word) > 0:
                tokenized_sentence.extend(tokenized_word)
                start.append(1)
                start.extend([0] * (len(tokenized_word) - 1))
        starts.append(start)
        if len(tokenized_sentence) > max_len - 2:
            print('Warning : Size', len(tokenized_sentence), ' is bigger, truncating index', i)
            tokenized_sentence = tokenized_sentence[:max_len - 2]
        lens.append(len(tokenized_sentence))
        # print('len is ', lens[-1])
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        tokenized_sentence.extend(["[PAD]"] * (max_len - len(tokenized_sentence)))
        tokenized_sentence_id = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokenized_sentences.append(tokenized_sentence)
        tokenized_sentences_ids.append(tokenized_sentence_id)
        tag_ids.append(tag_rev2[tags[i]]-1)
    np_input = np.array(tokenized_sentences_ids, dtype='long')
    np_tags = np.array(tag_ids, dtype='long')
    return np_input, np.array(lens, dtype='long'), tokenized_sentences, np_tags, starts

def join_bpe_split_tokens(tokens, original_sentence, starts):
    """

    joins splitted tokens using information stores in starts

    :param tokens: tokens given by BERT/RoBERTa tokenizer
    :param label_indices: labels indices to match to new tokens
    :param dict2: a dictionary mapping from indices to labels
    :param original_sentence: original text but splitted the same way tokenizer splits , needed because tokenizer puts some [UNK] sme times
    :param starts: starts[i]=1 is position i is start of a new word o.w. 0
    :return: new_tokens, new_labels, final_tokens
    """

    new_tokens = []
    final_tokens = []
    for token, orig_tok, start in zip(tokens, original_sentence, starts):
        if start == 0:
            if token.startswith('##'):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_tokens[-1] = new_tokens[-1] + token
        else:
            new_tokens.append(token)
            final_tokens.append(orig_tok)
    return new_tokens, final_tokens


class transformertagger():
    """
    This class helps to run inference on models

    """

    def __init__(self, model_path, model_class, tag2, device=torch.device("cuda")):
        """

        Initializes the class, loads model params and retains it

        :param model_path: the path to load the saved model, model is saved by tagger_trainer
        :param model_class: class of model
        :param device: torch.device, defaults to torch.device("cuda")
        """
        self.model = model_class.from_pretrained(model_path,
                                        tag_label_lst=[*tag2.keys()])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.model_name)
        self.device = device
        if self.device.type == "cuda":
            self.model.cuda()
        # Put the model into evaluation mode
        self.model.eval()

    def get_label(self, seqs, need_tokenization, bs=32):
        """

        This method returns labels for sequences (seqs)

        :param seqs: list of strings, input sentences
        :param bs: batch size to use in inference defaults to 32
        :return: final_toks, final_labels, final_toks in tokenized version of input sentences and final_labels is corresponding lables
        """
        input_ids, lens, tokenized, original_sentences, starts = tokenize_and_pad_text(seqs, self.tokenizer, need_tokenization, 65)
        attention_masks = [[i < lens[j] + 2 for i in range(len(ii))] for j, ii in enumerate(input_ids)]
        input_ids = input_ids.astype('int64')

        val_inputs = torch.tensor(input_ids)
        val_masks = torch.tensor(attention_masks)

        valid_data = TensorDataset(val_inputs, val_masks)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        predictions = []
        for batch in valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)
            # Move logits and labels to CPU
            logits = outputs[1][0]

            if type(logits) == list:
                logits = np.array(logits)
            else:
                logits = logits.detach().cpu().numpy()
            predictions.extend(np.argmax(logits, axis=1))

        final_toks = []
        for i in range(len(tokenized)):
            toks, final = join_bpe_split_tokens(tokenized[i][1:lens[i] + 1], original_sentences[i], starts[i])
            final_toks.append(final)

        final_tags = []
        for i in range(len(predictions)):
            final_tags.append(self.model.config.tag2[str(predictions[i]+1)])

        return final_toks, final_tags


def test_printer(data_path, toks, true_tags, predicted_tags):
    for sample_tokens, sample_tag, sample_predicted_tag in zip(toks, true_tags, predicted_tags):
        sample = ""
        for token in sample_tokens:
                sample += token + " "
        if sample_tag == sample_predicted_tag:
            sample += " <=> " + str(sample_tag) + " <=> " + str(sample_predicted_tag)
        else:
            sample += " <=> " + str(sample_tag) + " <=> " + str(sample_predicted_tag)
        f = open(data_path, "a")
        f.write(sample+"\n")
        f.close()

    f = open(data_path, "a")
    f.write(sample+"\n\n\n")
    f.close()
