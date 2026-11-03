"""使用冻结的中文 BERT 特征训练评论二分类层。"""
from pathlib import Path

import torch
from ai_starter import get_logger
from transformers import BertModel, BertTokenizer

log = get_logger(__name__)


class PostProcessingModel(torch.nn.Module):
    """先用预训练 BERT 提取评论特征，再用全连接层判断正面或负面。"""

    @staticmethod
    def load_tokenizer(model_path: Path | str) -> BertTokenizer:
        """从本地加载词表和分词器配置，不向 Hugging Face 发起下载请求。"""
        # 分词器和 BERT 使用同一模型目录，确保词元编号与预训练词表一致。
        log.info("加载本地分词器：%s", model_path)
        return BertTokenizer.from_pretrained(str(model_path), local_files_only=True)

    @staticmethod
    def create_model(model_path: Path | str, device: torch.device) -> "PostProcessingModel":
        """创建冻结 BERT 与分类层组成的模型，并整体移动到目标设备。"""
        # 构造函数仅加载本地权重；.to(device) 同时移动 BERT 和分类层。
        log.info("加载本地预训练模型：%s", model_path)
        return PostProcessingModel(model_path).to(device)

    def __init__(self, model_path: Path | str):
        """加载本地 BERT 并创建二分类层，不在模块导入时分配模型资源。"""
        super().__init__()
        # 注册为模型成员后，外层 .to(device) 才能统一移动 BERT 和分类层。
        # local_files_only=True 保证训练期间不会自动联网下载模型。
        self.pretrained_model = BertModel.from_pretrained(str(model_path), local_files_only=True)
        # 冻结预训练参数，只让新建的分类层参与梯度计算和训练。
        self.pretrained_model.requires_grad_(False)
        # 评估模式关闭 BERT 的 dropout，固定特征提取过程的行为。
        self.pretrained_model.eval()
        # 标准中文 BERT 的隐藏维度为 768；读取配置避免将维度写死。
        # 输出维度 2 分别对应负面和正面评论的原始分类分数。
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, 2)

    def train(self, mode: bool = True):
        """分类层跟随训练模式，冻结的 BERT 始终保持评估模式。"""
        # 父类会递归设置所有子模块模式，因此随后需要重新将 BERT 设为 eval。
        super().train(mode)
        self.pretrained_model.eval()
        return self

    def forward(self, input_ids, attention_mask, token_type_ids):
        """接收分词器张量，返回形状为 [批大小, 2] 的分类分数。"""
        # input_ids 是词元编号，attention_mask 标记有效位置，token_type_ids 标记句段。
        # 无梯度上下文只包住 BERT；后面的分类层仍需建立计算图以便训练。
        with torch.no_grad():
            outputs = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        # last_hidden_state 形状为 [批大小, 序列长度, 隐藏维度]。
        # 取第 0 个位置的 [CLS] 向量作为整条评论的特征，再送入分类层。
        return self.classifier(outputs.last_hidden_state[:, 0])
