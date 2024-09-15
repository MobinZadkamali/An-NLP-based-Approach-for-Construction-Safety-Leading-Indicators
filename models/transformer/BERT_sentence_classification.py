from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torch import nn
from models.transformer.classifier import Classifier

""" Using Bert with Self_Attention"""

class Bert(BertPreTrainedModel):
    def __init__(self, config, tag_label_lst):
        super().__init__(config)

        self.num_labels = len(tag_label_lst)

        self.bert = BertModel(config=config)  # Load pretrained bert
        self.self_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads)
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.final_classifier = Classifier(config.hidden_size, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            tags=None,
            output_attentions=True,
            lens=None,
            device=None
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True
        )
        sequence_output = outputs[0]

        # Apply self-attention
        attn_output, _ = self.self_attention(sequence_output, sequence_output, sequence_output)

        # Combine with original sequence output
        # combined_output = sequence_output + attn_output

        logits = self.final_classifier(attn_output[:, 0, :])


        total_loss = 0
        # Intent Softmax
        if tags is not None:
            loss_fct = nn.CrossEntropyLoss()
            total_loss += loss_fct(logits.view(-1, self.num_labels), tags.view(-1))


        outputs = ((logits, ),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)  # Logits is a tuple of intent and slot logits