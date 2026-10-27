from transformers import BertTokenizer,BertForSequenceClassification,pipeline

#设置具体包含config.json的目录，只支持绝对路径
model_dir = r"D:\jukeai\demo_03\trsanformers_test\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

#加载模型和分词器
# model = BertForSequenceClassification.from_pretrained("bert-base-chinese",cache_dir=model_dir)
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",cache_dir=model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

#创建分类pipleine
classifier = pipeline("text-classification",model=model,tokenizer=tokenizer,device="cuda")

#进行文本分类
result = classifier("你好，我是一款语言模型")
print(result)

print(model)