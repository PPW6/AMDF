# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @QQ交流群：830200766
# @QQ个人：2463739729


from transformers import BertModel
import torch.nn as nn
from config import parsers
import torch

class BertNerModel(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)

        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        self.loss_fun = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(768, class_num)

    def forward(self, batch_text, batch_label):  #
        texts, token_type_ids, attention_mask = batch_text.values()
        texts = texts.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        # Generate BERT embeddings
        bert_out = self.bert(input_ids=texts, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_out0 = bert_out[0]  # Shape: [batch_size, max_len, 768]

        # Predictions
        pred = self.classifier(bert_out0)  # Shape: [batch_size, max_len, class_num]

        if batch_label is not None:
            active_loss = (batch_label.view(-1) != -100)  # Ignore subwords or invalid labels (-100)

            # Apply mask to logits and labels for loss calculation
            active_logits = pred.view(-1, pred.shape[-1])[active_loss]
            active_labels = batch_label.view(-1)[active_loss]

            # Calculate loss only for valid tokens
            loss = self.loss_fun(active_logits, active_labels)

            # Mask invalid token predictions during evaluation
            valid_predictions = torch.argmax(pred, dim=-1)  # [batch_size, max_len]
            valid_predictions = valid_predictions.cpu().numpy()  # Convert to numpy

            # Mask out -100 positions
            masked_predictions = []
            for i, seq in enumerate(valid_predictions):
                # Only keep valid predictions where the batch_label is not -100
                masked_seq = [pred for pred, label in zip(seq, batch_label[i].cpu().numpy()) if label != -100]
                masked_predictions.append(masked_seq)

            return loss, masked_predictions
        else:
            # In inference mode, just apply the mask without calculating the loss
            valid_predictions = torch.argmax(pred, dim=-1)  # [batch_size, max_len]
            valid_predictions = valid_predictions.cpu().numpy()  # Convert to numpy

            # Mask out -100 positions
            masked_predictions = []
            for i, seq in enumerate(valid_predictions):
                masked_seq = [pred for pred, label in zip(seq, batch_label[i].cpu().numpy()) if label != -100]
                masked_predictions.append(masked_seq)

            return masked_predictions


