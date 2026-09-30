
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoTokenizer
from utils import get_idx2tag


class SentenceRE(nn.Module):
    def __init__(self, hparams):
        super(SentenceRE, self).__init__()
        self.bert_model = AutoModel.from_pretrained(hparams.bert_path)
        tokenizer = AutoTokenizer.from_pretrained(hparams.re_path)
        self.bert_model.resize_token_embeddings(len(tokenizer))
        for param in self.bert_model.parameters():
            param.requires_grad = True  # 能微调模型（训练时参数能变化），False->参数固定

        self.dense = nn.Linear(hparams.embedding_dim, hparams.embedding_dim)
        self.drop = nn.Dropout(hparams.dropout)
        self.activation = nn.Tanh()
        ##R-BERT
        # self.norm = nn.LayerNorm(hparams.embedding_dim * 3)
        # self.hidden2tag = nn.Linear(hparams.embedding_dim * 3, len(get_idx2tag(hparams.target_file)))
        #MTB
        self.norm = nn.LayerNorm(hparams.embedding_dim * 2)
        self.hidden2tag = nn.Linear(hparams.embedding_dim * 2, len(get_idx2tag(hparams.target_file)))

    # def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
    #######
    # R-BERT ： cls+ e1mask+e2mask
    ######
    #     # sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids,
    #     #                                                  attention_mask=attention_mask, return_dict=False)
    #     outputs = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids,
    #                                                      attention_mask=attention_mask, return_dict=False)
    #     sequence_output = outputs[0]
    #     cls_output = sequence_output[:, 0]
    #     # 每个实体的所有token向量的平均值
    #     e1_h = self.entity_average(sequence_output, e1_mask)
    #     e2_h = self.entity_average(sequence_output, e2_mask)
    #     e1_h = self.activation(self.dense(e1_h))
    #     e2_h = self.activation(self.dense(e2_h))
    #
    #     # [cls] + 实体1 + 实体2
    #     concat_h = torch.cat([cls_output, e1_h, e2_h], dim=-1)
    #     # concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
    #     concat_h = self.norm(concat_h)
    #     logits = self.hidden2tag(self.drop(concat_h))
    #
    #     return logits

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask,e1_pos, e2_pos):
        #######
        # MTB ： e1pos+e2pos
        ######
        outputs = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids,
                                                         attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]
        # 每个实体的所有token向量的平均值
        batch_size = sequence_output.size(0)

        batch_idx = torch.arange(batch_size,device=sequence_output.device)

        e1_h = sequence_output[batch_idx,e1_pos]

        e2_h = sequence_output[batch_idx,e2_pos]

        # [cls] + 实体1 + 实体2
        concat_h = torch.cat([e1_h, e2_h], dim=-1)
        # concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.drop(concat_h))

        return logits

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
