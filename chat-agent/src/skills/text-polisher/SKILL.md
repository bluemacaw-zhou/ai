---
name: text-polisher
description: 清理并润色用户提供的中英文文本，输出简洁、专业的版本。
when_to_use: 顶层工作流将“文本润色”选为细节子任务后，加载此 Skill。
allowed-tools: ""
---

# 文本润色

1. 先通过 `get_script` 读取 `normalize_text.py`。
2. 保留事实、名称、数字和原始语言。
3. 依据脚本规则删除冗余空白，将公告改写为简洁、专业的表达。
4. 只返回润色后的文本。
