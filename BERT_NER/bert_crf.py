from torch import nn
from torchcrf import CRF
from transformers import BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
from config import parsers

class Bert_CRF(nn.Module):
    def __init__(self, class_num, device):

        super(Bert_CRF, self).__init__()
        self.device = device
        # bert层
        self.bert = BertModel.from_pretrained(parsers().bert_pred)
        # dense层
        self.dense = nn.Linear(in_features=768, out_features=class_num)  # Adjust out_features
        # CRF层
        batch_first: bool = False
        self.crf = CRF(num_tags=class_num,batch_first=batch_first)
        # Adjust CRF transition scores for BIO tagging scheme
        start_transitions = self.crf.start_transitions.clone().detach()
        # print(start_transitions)
        transitions = self.crf.transitions.clone().detach()
        assert class_num % 2 == 1
        num_uniq_labels = (class_num - 1) // 2
        for i in range(num_uniq_labels, 2 * num_uniq_labels):
            start_transitions[i] = -10000
            for j in range(0, class_num):
                if j == i or j + num_uniq_labels == i: continue
                transitions[j, i] = -10000
        self.crf.start_transitions = nn.Parameter(start_transitions)
        self.crf.transitions = nn.Parameter(transitions)
        # 隐藏层
        self.hidden = None


    def forward(self, batch_text, batch_label):
        """
        token_texts:{"input_size": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                    "token_type_ids": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                     "attention_mask": tensor  [batch, 1, seq_len]->[batch, seq_len]->[seq_len, batch]
                     }
        tags:  [batch, seq_len]->[seq_len, batch]
        bert_out:  [batch, seq_len, hidden_size(768)]->[seq_len, batch, hidden_size]
        feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """
        texts, token_type_ids, masks = batch_text.values()
        texts = texts.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        masks = masks.squeeze(1)

        bert_out = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)[0]
        bert_out = bert_out.permute(1, 0, 2)
        feats = self.dense(bert_out)

        masks = masks.permute(1, 0)  # Permute to match CRF input shape
        masks = masks.clone().detach().bool()

        if batch_label is not None:
            batch_label = batch_label.permute(1, 0)  # Permute to match CRF input shape

            loss = -1 * self.crf(emissions=feats, tags=batch_label, mask=masks, reduction='mean')
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return loss, predictions
        else:
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return predictions


class BIO_Tag_CRF(CRF):
    def __init__(self, num_tags: int, device, batch_first: bool = False):
        super(BIO_Tag_CRF, self).__init__(num_tags=num_tags, batch_first=batch_first)
        self.device = device
        start_transitions = self.start_transitions.clone().detach()
        # print(start_transitions)
        transitions = self.transitions.clone().detach()
        assert num_tags % 2 == 1
        num_uniq_labels = (num_tags - 1) // 2
        for i in range(num_uniq_labels, 2 * num_uniq_labels):
            start_transitions[i] = -10000
            for j in range(0, num_tags):
                if j == i or j + num_uniq_labels == i: continue
                transitions[j, i] = -10000
        self.start_transitions = nn.Parameter(start_transitions)
        self.transitions = nn.Parameter(transitions)
        self.dense = nn.Linear(in_features=768, out_features=num_tags)

    def forward(self, logits, labels, masks, predit):

        new_logits, new_labels, new_attention_mask = [], [], []
        for logit, label, mask in zip(logits, labels, masks):
            new_logits.append(logit[mask])
            new_labels.append(label[mask])
            # new_attention_mask.append(torch.ones(new_labels[-1].shape[0], dtype=torch.uint8, device=self.device))
            new_attention_mask.append(torch.ones(new_labels[-1].shape[0], dtype=torch.bool, device=self.device))

        padded_logits = pad_sequence(new_logits, batch_first=True, padding_value=0)
        padded_logits = self.dense(padded_logits)
        padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=0)
        padded_attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        if predit==None:
            loss = -super(BIO_Tag_CRF, self).forward(emissions=padded_logits, tags=padded_labels,
                                                 mask=padded_attention_mask, reduction='mean')
            predictions = self.decode(emissions=padded_logits, mask=padded_attention_mask)
            return loss, predictions
        else:
            out = self.decode(emissions=padded_logits, mask=padded_attention_mask)
            # assert (len(out) == len(labels))
            # out_logits = torch.zeros_like(logits)
            # for i in range(len(out)):
            #     k = 0
            #     for j in range(len(labels[i])):
            #         if labels[i][j] == -100: continue
            #         out_logits[i][j][out[i][k]] = 1.0
            #         k += 1
            #     assert (k == len(out[i]))
            return out
class Bert_CRF_SPLIT(nn.Module):
    def __init__(self, class_num, device):
        super(Bert_CRF_SPLIT, self).__init__()
        self.device = device
        # bert层
        self.bert = BertModel.from_pretrained(parsers().bert_pred)
        # CRF层
        self.crf_ = BIO_Tag_CRF(class_num, device, batch_first=True)

    def forward(self, batch_text, batch_label, predit=None):
        """
        token_texts:{"input_size": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                    "token_type_ids": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                     "attention_mask": tensor  [batch, 1, seq_len]->[batch, seq_len]->[seq_len, batch]
                     }
        tags:  [batch, seq_len]->[seq_len, batch]
        bert_out:  [batch, seq_len, hidden_size(768)]->[seq_len, batch, hidden_size]
        feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """
        texts, token_type_ids, attention_mask = batch_text.values()
        texts = texts.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        outputs = self.bert(input_ids=texts, attention_mask=attention_mask, token_type_ids=token_type_ids)

        bert_out = outputs[0]

        # Create a label mask where -100 positions are marked as False
        valid_label_mask = (batch_label != -100)
        return self.crf_(bert_out, batch_label, valid_label_mask, predit)

