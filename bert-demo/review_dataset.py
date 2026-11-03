from functools import partial
from pathlib import Path

import torch
from transformers import BertTokenizer

from ai_starter import get_logger
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

log = get_logger(__name__)

class ReviewDataset(Dataset):
    #初始化数据集
    def __init__(self,split):
        #从磁盘加载数据
        self.dataset = load_from_disk(str(Path(__file__).resolve().parent / "data" / "ChnSentiCorp"))
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        else:
            log.info("数据名错误！")

    #返回数据集长度
    def __len__(self):
        return len(self.dataset)

    #对每条数据单独做处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text,label

    @staticmethod
    def encode_batch(samples, tokenizer: BertTokenizer, maximum_length: int):
        """将一批评论转换为模型输入和类别标签。"""
        sentences = [sample[0] for sample in samples]

        # CrossEntropyLoss 接收整型类别编号：0 为负面评论，1 为正面评论。
        labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)

        # 批量编码时自动添加 [CLS]、[SEP] 等特殊标记。
        # truncation=True 截断超长评论；padding="max_length" 将短评论补齐到统一长度。
        # return_tensors="pt" 直接返回 PyTorch 张量，便于送入模型。
        encoded_inputs = tokenizer(
            sentences, truncation=True, max_length=maximum_length,
            padding="max_length", return_tensors="pt",
        )

        # input_ids：词元编号；attention_mask：区分真实词元与补齐位置。
        # token_type_ids：句段编号，当前每条样本只有一段评论，通常全部为 0。
        return dict(encoded_inputs), labels


    @staticmethod
    def create_data_loader(
        split: str, tokenizer: BertTokenizer, batch_size: int, maximum_length: int = 512
    ) -> DataLoader:
        """加载指定数据划分，并配置批量读取与编码过程。"""
        dataset = ReviewDataset(split)
        if len(dataset) == 0:
            raise ValueError(f"数据划分为空：{split}")
        log.info("加载 %s 数据：%s 条", split, len(dataset))

        # 训练集打乱顺序；验证集保持固定顺序，方便重复比较。
        # drop_last=False 保留最后不足一批的样本，验证指标覆盖完整验证集。
        # partial 预先绑定分词器和长度，DataLoader 调用时只需传入本批样本。
        return DataLoader(
            dataset, batch_size=batch_size,
            shuffle=split == "train", drop_last=False,
            collate_fn=partial(ReviewDataset.encode_batch, tokenizer=tokenizer, maximum_length=maximum_length),
        )


if __name__ == '__main__':
    dataset = ReviewDataset("train")
    for data in dataset:
        log.info(data)
