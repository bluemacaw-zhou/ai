# Harness：可刷新数据与动态 A2UI 渲染设计

本文档从静态到动态组织：先定义基础数据结构，再定义底层 Agent，随后定义 Harness 主 Agent，最后给出动态时序。

## 基础数据结构

1. `01-render-command-input-model.puml`：Render 类型 `CommandRecord.input` 的固定边界：`requirement`、`sql`、`data`、`meta_data`；同时说明 SQL → surfaceId 映射。
2. `02-harness-state-model.puml`：HarnessState、CommandRecord、RenderRegistry 的持久化关系与状态约束。

## 底层 Agent 流程

3. `03-data-query-agent-langgraph.puml`：DataQueryAgent 的 LangGraph 编排；主体/板块识别、Cosmos 查询与数据/文档兜底。
4. `04-render-tool-internal-flow.puml`：Render Tool 按 SQL 映射创建新 UI 节点或更新既有节点的内部状态机。

## Harness 主 Agent 流程

5. `05-harness-loop-overview.puml`：当前 pending 命令的执行、工具调用、ToolMessage 登记、本轮评审、呈现判断、下一命令决策与任务终止判定（原图 05、06 已合并为一张图）。

## 动态时序

6. `06-initialization-and-progress-sequence.puml`：ToolMessage 登记后追加下一条命令，并由固定代码节点更新“正在执行”状态栏的时序。
7. `07-final-result-sequence.puml`：render 命令按 SQL 创建或更新确定 UI，并回写 HarnessState、发布 surface 的时序。

## Harness 外层入口

8. `08-input-preprocessing-and-routing.puml`：外层 LLM 对用户问题进行最小预处理；金融信息获取问题规范化后进入 Harness，其它问题由通用 LLM 直接回答。

关键原则：每个 A2A task 对应一个 HarnessState，其中 CommandList 是可回放的业务状态。05 执行唯一的最后一条 `pending CommandRecord`；每次工具返回 ToolMessage 后进入登记与评审环节。评审成功且跳过渲染时，先由 `CheckGoalCompletion` 判断根问题 `original_question` 是否已被完整解答——是则直接返回 `route=normal_completed`，这是唯一的正常完成出口；否则才进入 `DecideNextGoal` 决定是否还有下一步。评审失败时，先由 `CheckFailureThresholds`（合并了长度上限与全局失败堆积判定的代码节点）做快速止损：命中长度上限直接降级；全局失败总数达到 `failure_streak_threshold` 阈值（不要求连续）则交给 `CheckSameGoalRepetition` 做语义判断，是同一目标反复失败则降级，否则放行进入 `DecideNextGoal`。`DecideNextGoal` 返回 `no_next_command` 现在等价于"根目标无法达成"，直接降级为 `goal_unreachable`，不再被当作正常完成的充分条件。四种降级原因（`no_tool_matched`、`max_commands_exceeded`、`repeated_goal_failure`、`goal_unreachable`）统一走 `route=fallback_required`。DataQueryAgent 的固定编排见图 03；Render 输入由图 01 固定；Render Tool 在图 04 中使用 HarnessState 的规范 SQL → surfaceId 映射，创建或更新确定的 UI 目标，并回写最新 A2UI surface。
