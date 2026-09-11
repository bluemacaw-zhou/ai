---
name: announcement-workflow
description: 为产品公告相关请求设计顶层执行流程，并确定后续需要加载的细节 Skill。
when_to_use: 用户要求优化、审阅或撰写产品公告时，优先加载此 Skill。
allowed-tools: ""
---

# 产品公告工作流

对于产品公告润色请求，下一步加载 `text-polisher` Skill。它负责细节写作规则和文本规范化脚本。在该 Skill 加载完成前，不得直接润色文本。
