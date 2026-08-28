"""服务使用的 LangGraph 图集合。

当前体系：

* :mod:`graph.complete_task`：``CompleteTask``，通用的任务收尾节点（推送
  TaskUpdater completed 事件并产出 route），不绑定任何具体子图的 State
  类型，供任意子图直接复用。当前由 05 图使用。
* :mod:`graph.main_graph`：``MainGraph``，08 图（design/harness/
  08-input-preprocessing-and-routing.puml）的完整实现，是当前系统的顶层
  入口图，完整 7 个节点：ReceiveUserQuestion / RewriteQuestion /
  PreprocessQuestion / StartHarness / AnswerFallback / AnswerDirectly /
  CreateMarkdownSurfaceAndSend。
* :mod:`graph.harness_graph`：``HarnessGraph``，05 图（design/harness/
  05-harness-loop-overview.puml）的简化实现，当前只有 ``ExecuteCommand``
  （真实 LLM + MCP 工具执行节点）与 ``CompleteTask``（来自
  ``graph.complete_task``）两个节点，由 08 图的 ``StartHarness`` 节点调用。

已归档、放弃维护：

* :mod:`graph.legacy_main_graph`：旧的 ``MainGraph``/``MainGraphFactory``/
  ``DataGraph``/``RenderGraph``，整体搬迁保留但不再被生产代码引用，也不再
  同步维护其内部的 import 路径。

本文件不做子包聚合导出（不像旧版本那样在这里 import DataGraph/RenderGraph
等），避免顶层 ``graph`` 包的导入触发所有子包加载；调用方应直接从具体子包
（``graph.complete_task`` / ``graph.main_graph`` / ``graph.harness_graph``）
导入需要的类。
"""
