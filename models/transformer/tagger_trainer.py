import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from .seqTagger import get_ouputs
from tqdm import tqdm



def run_transformer_trainer(Data, batch_size, FULL_FINETUNING, model, tokenizer,
                            device, validation_func=None,
                            lr=5e-5, epochs=10, save_dir=None):
    '''
      train a transformer tagger

      :param nparray input_ids: ids tokens, given by tokenizer
      :param nparray tags: ids of tags
      :param list starts: identifies if the position is start of a token, starts[i] 1 is it is start o.w. 0
      :param int batch_size: model training batch_size
      :param intlist lens: length of sequences in input_ids
      :param Boolean FULL_FINETUNING: if set to true, update BERT (or any language model) parameters too, if set to false, only upates classifier's parameters
      :param tokenizer tokenizer: needed when saving the model
      :param intlist val_idnxs: indices of input_ids to consider for validation part
      :param intlist test_idnxs: indices of input_ids to consider for test part
      :param function validation_func: function to log proper metric in each epoch, takes four args valid_true_labels, valid_predicted_labels, test_true_labels,
                   test_predicted_labels
      :param float lr: learning rate
      :param int epochs: training iterations
      :param string save_dir: directory to save the model
      '''

    tr_inputs = Data["tr_inputs"]
    tr_masks = Data["tr_masks"]
    tr_tags = Data["tr_tags"]

    val_inputs = Data["val_inputs"]
    val_masks = Data["val_masks"]
    val_tags = Data["val_tags"]

    test_inputs = Data["test_inputs"]
    test_masks = Data["test_masks"]
    test_tags = Data["test_tags"]

    tr_lens = Data["tr_lens"]
    tr_starts = Data["tr_starts"]
    val_lens = Data["val_lens"]
    val_starts = Data["val_starts"]
    test_lens = Data["test_lens"]
    test_starts = Data["test_starts"]

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, torch.Tensor(tr_lens))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags, torch.Tensor(val_lens))
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_tags, torch.Tensor(test_lens))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    if device.type == "cuda":
        # print('Its CUDA')
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    # no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8
        # ,weight_decay=1e-6, amsgrad=True
    )

    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values, f1_scores, f1_scores_val = [], [], [], []
    for epoch in range(epochs):
        print('Epoch', epoch)
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0
        # Training loop
        train_dataloader = tqdm(train_dataloader, mininterval=60)
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_tags, lens = batch
            # try:
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, tags=b_tags, lens=lens, device=device)

            # get the loss
            loss = outputs[0]

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        tr_true_tags, tr_pred_tags = get_predictions(model, train_dataloader, device, model.config.tag2)

        val_true_tags, val_pred_tags = get_predictions(model, valid_dataloader, device, model.config.tag2)

        test_true_tags, test_pred_tags = get_predictions(model, test_dataloader, device, model.config.tag2)
        if validation_func is not None:
            f1_score, f1_score_val = validation_func(tr_true_tags, tr_pred_tags,
                                                     val_true_tags, val_pred_tags,
                                                     test_true_tags, test_pred_tags)
            f1_scores.append(f1_score)
            f1_scores_val.append(f1_score_val)

    acc = accuracy_score(test_true_tags, test_pred_tags)
    print("Metrics on Validation/Test data: Acc={0:.5f}".format(acc))
    print(classification_report(test_true_tags, test_pred_tags))
    print("Validatin/Test Confusion Matrix:")
    print(confusion_matrix(test_true_tags, test_pred_tags))
    print("Test Accuracy: ",accuracy_score(test_true_tags, test_pred_tags))
    print("Test ROC_AUC Score: ",roc_auc_score(test_true_tags, test_pred_tags))
    
    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(save_dir)
        save_contiguous_model(model_to_save, save_dir)
        tokenizer.save_pretrained(save_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(save_dir, 'training_args.bin'))


def save_contiguous_model(model, save_directory):
    for param in model.parameters():
        param.data = param.data.contiguous()  # Make each parameter tensor contiguous
    model.save_pretrained(save_directory)


def get_predictions(model, dataloader, device, tag2):
    predictions_tags, true_tags = get_ouputs(model, dataloader, device)
    predicted_labels_tags = []
    true_labels_tags = []
    for i in range(len(predictions_tags)):
        predicted_labels_tags.append(tag2[predictions_tags[i]+1])
        true_labels_tags.append(tag2[true_tags[i]+1])
    return predicted_labels_tags, true_labels_tags



def validation(tr_true_tags, tr_pred_tags,
               val_true_tags, val_pred_tags,
               test_true_tags, test_pred_tags):
    
    # Train Evaluation
    tracc = accuracy_score(tr_true_tags, tr_pred_tags)
    print("Metrics on Train data: Acc={0:.5f}".format(tracc))
    print(classification_report(tr_true_tags, tr_pred_tags))
    print("Train Confusion Matrix:")
    print(confusion_matrix(tr_true_tags, tr_pred_tags))

    return f1_score(test_true_tags, test_pred_tags, average='weighted'), f1_score(val_true_tags, val_pred_tags, average='weighted')
