# src/models/bert.py
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertClassifier(BertPreTrainedModel):
    """基于BERT的欺诈分类器"""

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)  # 二分类
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取[CLS] token的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 分类
        logits = self.classifier(pooled_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits
