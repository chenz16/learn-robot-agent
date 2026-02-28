# MVP PRD: Robot Agent v0.1-alpha

## 1. Executive Summary

基于 [nanobot](https://github.com/HKUDS/nanobot)（~4,000 行 Python asyncio 框架）构建的机器人 agent，通过 CLI 接收自然语言指令，调度 VLM（视觉理解）和 VLA（动作生成）模型，在 LIBERO 仿真环境中控制机器人完成操作任务。核心理念：机器人动作和软件动作是同一种抽象——Tool。

**目标用户**: 机器人研究团队，需要一个轻量 agent 框架验证 "LLM 调度 VLA/VLM 执行操作任务" 的端到端可行性。

**核心 demo**: 两个 LIBERO-10 (Long) 多步任务，验证 **LLM 任务分解 + VLA 逐步执行**——
1. 重复型分解："put both the alphabet soup and the tomato sauce in the basket"（同一动作对不同物体执行两轮）
2. 链式型分解："put the black bowl in the bottom drawer of the cabinet and close it"（不同动作按顺序串联：开→放→关）

两个任务都需要 LLM 拆解为多个子任务，VLA 无法一步完成。Agent 自主完成 感知 → 推理 → 分步执行 → 验证 闭环，Franka Panda 机械臂在 LIBERO 仿真中成功完成操作。

---

## 2. Scope

### 2.1 IN（MVP 范围内）

| 维度 | 内容 |
|------|------|
| 机器人数量 | 单机器人 |
| 仿真环境 | LIBERO（MuJoCo），Franka Panda 固定底座，130+ 桌面操作任务 |
| 交互方式 | CLI 终端（文本输入） |
| Agent 框架 | nanobot，复用 AgentLoop、ToolRegistry、SkillsLoader 等 |
| 模型 | VLM（场景理解）+ VLA（动作生成）+ LLM（推理调度） |
| 控制闭环 | PRAE：Prepare → Perceive → Reason → Act → Evaluate |
| Robot Tools | 8 个 base tools |
| VLA 接入 | Mock adapter（开发调试）+ HTTP adapter（接真实 VLA 推理服务） |
| 安全 | 紧急停止 + 速度限制 |
| 运行模式 | Mock 模式（无 GPU）+ LIBERO 仿真模式（CPU） |

### 2.2 OUT（不在 MVP 范围）

| 功能 | 推迟原因 |
|------|---------|
| 多机器人协调 | 单机器人已足够验证架构 |
| 语音输入（板载麦克风 / Web Voice） | 需要 ASR/VAD 依赖，CLI 足够 |
| LoRA 热切换 | Base model 足够完成 demo 任务 |
| 模型生命周期管理（加载/卸载/VRAM 管理） | 模型由外部手动启动 |
| OTA 更新 / 模型维护 | 运维层面，非架构验证 |
| 自主巡逻 / 任务自动领取 | 应用层行为，后续用 SKILL.md 实现 |
| 仿真 fork-compare-promote | 需要多并行仿真实例 |
| Web UI / 多通道（Telegram、Slack 等） | CLI 足够，nanobot channels 后续零改动启用 |
| 安全区域围栏 / 审计日志 | 需要 3D 空间感知，MVP 暂不需要 |
| 真实硬件（Unitree G1 + GR00T） | MVP 先在仿真中验证 |
| 导航（移动底座） | LIBERO 为固定底座，导航需 RoboCasa 等支持移动底座的仿真 |

---

## 3. Architecture Overview

### 3.1 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  User (CLI Terminal)                  │
└──────────────────────┬──────────────────────────────┘
                       │ 自然语言指令
                       ▼
┌─────────────────────────────────────────────────────┐
│                Agent Core (nanobot)                   │
│                                                       │
│  AgentLoop ──► ToolRegistry ──► Robot Tools           │
│      │              │                                 │
│      │         ┌────┴────────────────────────┐       │
│      │         │ look, move, grasp, perceive │       │
│      │         │ start_subtask, check_loops  │       │
│      │         │ wait_subtask, emergency_stop│       │
│      │         └────┬──────────┬─────────────┘       │
│      │              │          │                      │
│  ContextBuilder  MemoryStore  SkillsLoader            │
└──────┬──────────────┼──────────┼─────────────────────┘
       │              │          │
       ▼              ▼          ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│ LLM      │  │ Loop Manager │  │ VLA Adapter  │
│ Provider │  │              │  │              │
│          │  │ VLA @ 2-20Hz │  │ Mock / HTTP  │
│ Local    │  │ Terminators  │  │              │
│ vLLM     │  └──────┬───────┘  └──────┬───────┘
└──────────┘         │                 │
              ┌──────┴─────────────────┴──────┐
              │        LIBERO (MuJoCo)         │
              │  130+ manipulation tasks       │
              │  Gymnasium API                 │
              └────────────────────────────────┘
```

### 3.2 三层抽象

Agent 的推理和控制分为三个逻辑层，频率递增、抽象递减：

| 层 | 职责 | 典型延迟 | MVP 实现 |
|----|------|---------|---------|
| **Intention** | 意图识别、槽位提取、判断是否需要完整规划 | 50-800ms | LLM 单轮推理 |
| **Cognition** | 场景感知 + 任务规划，输出结构化 plan | 0.5-10s | VLM 分析 + LLM 规划 |
| **Action** | 高频控制循环，VLA 预测 + 安全检查 + 执行 | 2-20Hz, < 20ms/step | LoopManager + VLA adapter |

设计约束：Action 层不可被 Intention/Cognition 阻塞——Cognition 的延迟只影响下一个子任务的决策，不中断正在执行的动作循环。

### 3.3 Nanobot 复用 vs 新增

| 组件 | 来源 | 说明 |
|------|------|------|
| AgentLoop | nanobot 复用 | 消息消费 → LLM 调用 → Tool 调度 → 保存会话 |
| ToolRegistry | nanobot 复用 | 动态注册/注销，JSON Schema 校验 |
| ContextBuilder | nanobot 复用 | 拼接 system prompt：identity + memory + skills |
| MemoryStore | nanobot 复用 | LLM 驱动的记忆压缩 |
| SkillsLoader | nanobot 复用 | SKILL.md 加载，YAML frontmatter |
| SubagentManager | nanobot 复用 | 后台 asyncio 子 agent，最多 15 轮迭代 |
| LLMProvider | nanobot 复用 | 支持 20+ 模型提供商 |
| **Robot Tools** | **新增** | look, move, grasp 等 8 个 |
| **VLA Adapters** | **新增** | Mock + HTTP 两种 |
| **LoopManager** | **新增** | asyncio 控制循环 + cognition handoff |
| **Terminators** | **新增** | StepLimit + PositionThreshold |
| **SafetyManager** | **新增** | E-Stop + 速度限制 |
| **RobotEnv** | **新增** | Mock + LIBERO 环境接口 |

**新增代码量估算**: ~800 行。其余全部复用 nanobot。

---

## 4. Functional Requirements

### FR-1: Agent Core

**需求**: 复用 nanobot 的 agent 核心，不做修改。

- AgentLoop 消费消息、调用 LLM、分发 Tool 调用、保存会话
- ToolRegistry 支持动态注册，所有 tool 参数经 JSON Schema 校验
- ContextBuilder 从 identity + memory + skills + bootstrap files 拼接 system prompt
- MemoryStore 在会话超过 `memory_window` 时自动压缩
- SkillsLoader 加载 `skills/` 目录下的 SKILL.md
- SubagentManager 支持 spawn 子 agent 执行感知/执行/评估子任务

**配置需求**:
- 指定工作目录、模型、最大迭代次数、记忆窗口、temperature
- 模型默认使用本地 vLLM，可选配置远程 LLM（用户自行配置 API）

### FR-2: Robot Tools

**需求**: 8 个 base tools，覆盖 PRAE 闭环的最小必要能力。

| Tool | 输入 | 输出 | 用途 |
|------|------|------|------|
| `look` | question: str | 场景描述 + 物体列表 | 捕获图像 + VLM 分析 |
| `move` | target: str, position: [x,y,z] | 执行状态 | 移动末端执行器到目标位置 |
| `grasp` | action: "open" \| "close" | 夹爪状态 | 控制夹爪 |
| `perceive` | goal: str | 深度感知结果 | 通过子 agent 进行多步观察分析 |
| `start_subtask` | instruction: str, target: dict | 子任务 ID + 状态 | 启动 VLA 控制循环 |
| `check_loops` | — | 循环状态 + 统计 | 查询当前控制循环运行状态 |
| `wait_subtask` | timeout: float | 完成状态 + 结果 | 阻塞等待子任务完成 |
| `emergency_stop` | — | 确认 | 立即停止所有机器人运动 |

**扩展机制**: 用户可通过 Python 代码注册自定义 tool（继承 Tool ABC，调用 `agent.tools.register()`）。自定义 tool 和 base tool 在 LLM 视角完全平等，ToolRegistry 是扁平的。

### FR-3: Model Management

**需求**: 极简模型管理——确认外部启动的模型服务是否就绪。

MVP 中模型服务（VLM、VLA）由开发者手动启动。Agent 只需：

| 能力 | 说明 |
|------|------|
| **健康检查** | 查询各模型服务是否在线、响应延迟 |
| **状态查询** | 当前加载了哪些模型、各服务地址 |
| **就绪确认** | PRAE 循环开始前，确认 VLM + VLA + Sim 全部 ready |

Agent 通过 `model_health` 和 `model_ensure` 两个内部工具完成上述能力。不涉及模型加载/卸载/VRAM 管理。

### FR-4: VLA Adapter Layer

**需求**: 统一的 VLA 接口抽象，MVP 提供两种实现。

VLA adapter 定义统一接口：输入（图像 + 指令 + 机器人状态）→ 输出（动作序列）。

| Adapter | 用途 | 后端 |
|---------|------|------|
| **MockVLAAdapter** | 开发调试，无需 GPU | 进程内模拟，返回随机/固定动作序列 |
| **HTTPVLAAdapter** | 接真实 VLA 推理服务 | HTTP POST 到 VLA server（SmolVLA / pi0 / 其他） |

接口契约：
- **predict**: 输入 observation dict + instruction string → 返回动作序列 (action_horizon × action_dim)
- **reset**: 重置内部状态（新 episode）
- **health_check**: 存活检测
- **get_action_horizon**: 返回每次预测的动作步数（chunking size）

通过配置切换 adapter 类型，对 LoopManager 和上层 agent 透明。

### FR-5: Multi-Frequency Control

**需求**: LoopManager 管理 VLA 控制循环，与上层推理解耦。

LoopManager 是 Action 层的执行引擎：

- **start_subtask**: 接收指令 + 目标 + 路由配置，启动异步控制循环
- **控制循环**: 以 action_hz 频率运行——获取观测 → VLA 预测 → 安全检查 → 执行动作 → 检查终止条件
- **wait_for_completion**: 阻塞等待子任务完成或超时
- **stop**: 强制停止所有活跃循环

设计约束：
- 控制循环运行在 asyncio task 中，不阻塞 agent 主循环
- VLA predict() 如果是阻塞调用，使用 `asyncio.to_thread()` 包装
- Cognition 层的延迟不影响正在执行的 Action 循环

### FR-6: Termination Strategies

**需求**: VLA 不会自行停止，需要外部终止策略。MVP 提供 2 种。

| 策略 | 触发条件 | 用途 |
|------|---------|------|
| **StepLimitTerminator** | 步数 >= max_steps | 安全兜底，防止无限运行 |
| **PositionThresholdTerminator** | 末端执行器距目标 < threshold | move-to-target 类任务的完成判定 |

两种策略可组合使用（AND / OR）。路由配置指定每个难度等级使用的终止策略组合。

### FR-7: Routing

**需求**: 任务难度决定控制参数。MVP 提供 1 个默认路由配置、2 个难度等级。

**难度分类**:
- **easy**: 已知物体 + 简单动词（"拿起红色杯子"）
- **hard**: 复合多步任务（"put the black bowl in the bottom drawer of the cabinet and close it"）

**路由决定的参数**:

| 参数 | easy | hard |
|------|------|------|
| action_hz | 10 | 20 |
| max_steps | 100 | 500 |
| position_threshold | 0.05m | 0.03m |
| 终止策略 | StepLimit + PositionThreshold | StepLimit + PositionThreshold |
| LLM 重规划 | 无 | 每个子任务后重新规划 |

路由配置存储为 YAML 文件，agent 在 PRAE 的 Reason 阶段根据指令判断难度并选择路由。

### FR-8: Safety

**需求**: 最基础的安全保障——能停、能限速。

| 安全层 | 行为 |
|--------|------|
| **E-Stop** | `emergency_stop` 触发后，立即停止所有运动，状态持久化（重启后仍生效），需手动解除 |
| **Velocity Limit** | 每个动作执行前检查速度，超过 max_velocity 自动钳制（clamp），不拒绝动作 |

安全检查位于 LoopManager 控制循环内，在 `env.step()` 之前执行。每次安全事件记录到日志。

---

## 5. PRAE Loop

两个验收任务都来自 **LIBERO-10 (Long)**，都需要 LLM 分解、VLA 逐步执行。

### 5.1 任务 A（重复型分解）：put both the alphabet soup and the tomato sauce in the basket

**LIBERO 任务 ID**: `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`

**LLM 分解要点**: 同一动作模式（抓取→放入）对不同物体执行两轮，每轮之间需重新感知。

```
Step 1: PREPARE
  Agent 收到用户指令 → model_ensure 确认 VLM + VLA + Sim 全部 ready
  LIBERO 加载 LIVING_ROOM_SCENE2

Step 2: PERCEIVE
  look("describe all objects on the table and their positions") →
    VLM 返回："alphabet soup at [0.2, -0.1, 0.10], tomato sauce at [0.4, 0.2, 0.10],
              basket at [0.6, 0.0, 0.05], ..."

Step 3: REASON (LLM 任务分解)
  LLM 分析指令 "put both ... in the basket" →
    识别：2 个目标物体，1 个目标容器
    分解为 2 轮，每轮 = pick + place：
      轮 1: pick up alphabet soup → place in basket
      轮 2: pick up tomato sauce → place in basket
    判断难度：hard（多物体复合任务）
    选择路由：default/hard (action_hz=20, max_steps=500)

=== 轮 1: alphabet soup ===

Step 4: ACT (子任务 1: 抓取 alphabet soup)
  start_subtask("pick up the alphabet soup",
                target={"object": "alphabet_soup", "position": [0.2, -0.1, 0.10]}) →
    VLA 控制循环 @ 20Hz → 接近 → 下降 → 闭合夹爪 → 抬起
    StepLimit(500) 终止

Step 5: ACT (子任务 2: 放入 basket)
  start_subtask("place the alphabet soup in the basket",
                target={"object": "basket", "position": [0.6, 0.0, 0.05]}) →
    VLA 控制循环 → 移到篮子上方 → 下降 → 松开夹爪
    StepLimit(500) 终止

Step 6: EVALUATE (中间检查)
  look("is the alphabet soup in the basket?") →
    VLM 确认："alphabet soup is inside the basket"
  通过 → 进入轮 2

=== 轮 2: tomato sauce ===

Step 7: PERCEIVE (重新感知——场景已变，soup 不在桌上了)
  look("where is the tomato sauce now?") →
    VLM 返回："tomato sauce at [0.4, 0.2, 0.10]"

Step 8: ACT (子任务 3: 抓取 tomato sauce)
  start_subtask("pick up the tomato sauce") →
    VLA 控制循环 → 抓取番茄酱

Step 9: ACT (子任务 4: 放入 basket)
  start_subtask("place the tomato sauce in the basket") →
    VLA 控制循环 → 放入篮子

Step 10: EVALUATE (最终)
  look("are both the alphabet soup and the tomato sauce in the basket?") →
    VLM 确认："both items are in the basket"
  成功 → 回复用户 "done: both items placed in the basket"

=== 失败处理 ===
  某轮 EVALUATE 失败 → 对该轮重新 PERCEIVE → ACT → EVALUATE
  单轮最多重试 3 次
  3 次失败 → 上报用户，说明卡在哪一轮（如 "failed to place tomato sauce, gripper did not close"）
```

### 5.2 任务 B（链式型分解）：put the black bowl in the bottom drawer of the cabinet and close it

**LIBERO 任务 ID**: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`

**LLM 分解要点**: 不同类型的动作按顺序串联（开抽屉 → 抓碗 → 放入 → 关抽屉），每个子任务的动作模式不同，且有依赖关系（必须先开抽屉才能放入）。

```
Step 1: PREPARE
  Agent 收到用户指令 → model_ensure 确认 ready
  LIBERO 加载 KITCHEN_SCENE4

Step 2: PERCEIVE
  look("describe the scene: where is the black bowl and the cabinet?") →
    VLM 返回："black bowl on the table at [0.3, 0.0, 0.08],
              cabinet with bottom drawer at [0.6, -0.2, 0.15], drawer is closed"

Step 3: REASON (LLM 任务分解)
  LLM 分析指令 "put the black bowl in the bottom drawer ... and close it" →
    识别依赖链：要放进抽屉 → 抽屉必须先打开 → 放入后必须关闭
    分解为 4 个顺序子任务：
      子任务 1: open the bottom drawer of the cabinet
      子任务 2: pick up the black bowl
      子任务 3: place the black bowl in the drawer
      子任务 4: close the bottom drawer
    判断难度：hard（4 步链式依赖）
    选择路由：default/hard (action_hz=20, max_steps=500)

Step 4: ACT (子任务 1: 开抽屉)
  start_subtask("open the bottom drawer of the cabinet",
                target={"object": "bottom_drawer", "position": [0.6, -0.2, 0.15]}) →
    VLA 控制循环 → 接近抽屉把手 → 抓握 → 向外拉
    StepLimit(500) 终止

Step 5: EVALUATE (中间检查 1)
  look("is the bottom drawer open?") →
    VLM 确认："the bottom drawer is open"
  通过 → 继续

Step 6: ACT (子任务 2: 抓碗)
  start_subtask("pick up the black bowl from the table") →
    VLA 控制循环 → 接近碗 → 下降 → 闭合夹爪 → 抬起

Step 7: ACT (子任务 3: 放入抽屉)
  start_subtask("place the black bowl in the open drawer") →
    VLA 控制循环 → 移到抽屉上方 → 下降到抽屉内 → 松开夹爪

Step 8: EVALUATE (中间检查 2)
  look("is the black bowl inside the drawer?") →
    VLM 确认："the black bowl is in the bottom drawer"
  通过 → 继续

Step 9: ACT (子任务 4: 关抽屉)
  start_subtask("close the bottom drawer of the cabinet") →
    VLA 控制循环 → 接近抽屉 → 向内推

Step 10: EVALUATE (最终)
  look("is the drawer closed with the bowl inside?") →
    VLM 确认："the bottom drawer is closed"
  成功 → 回复用户 "done: bowl placed in drawer and drawer closed"

=== 失败处理 ===
  子任务失败 → 根据当前状态决定回退策略：
    - 子任务 1 失败（开抽屉）→ 重试开抽屉
    - 子任务 3 失败（放入）→ 可能需要重新抓碗（从子任务 2 重试）
    - 子任务 4 失败（关抽屉）→ 重试关抽屉
  每个子任务最多重试 3 次
  3 次失败 → 上报用户，说明当前状态（如 "drawer is open, bowl is on the table, failed to pick up bowl"）
```

---

## 6. Non-Functional Requirements

### 6.1 Performance

| 指标 | 目标 |
|------|------|
| VLA 控制循环延迟 | < 20ms / step |
| VLM 感知延迟 | < 2s / frame |
| LLM 规划延迟 | < 10s / decision |
| 端到端任务（简单 pick-and-place） | < 60s |

### 6.2 Concurrency

- 全部使用 asyncio，不使用 threading.Lock
- 单事件循环处理所有协程
- VLA predict() 等阻塞调用使用 `asyncio.to_thread()`

### 6.3 Reliability

| 场景 | 行为 |
|------|------|
| VLA 服务不可达 | 重试 3 次，然后报告用户 |
| LLM API 超时 | 指数退避重试，降级到本地模型 |
| 子任务执行失败 | PRAE 重试（最多 3 次），3 次失败上报用户 |
| E-Stop 触发 | 所有运动立即停止，需手动解除 |

---

## 7. Verification Criteria

2 个 LIBERO-10 (Long) 任务，验证两种 LLM 分解模式 + VLA 逐步执行。

### 7.1 任务 A：重复型分解（Mock + LIBERO）

**LIBERO 任务**: `put both the alphabet soup and the tomato sauce in the basket` (LIVING_ROOM_SCENE2)

**验证重点**: LLM 能将 "both X and Y" 分解为 2 轮相同模式的 pick-and-place，每轮之间重新感知。

| 模式 | 通过条件 |
|------|---------|
| **Mock** | LLM 正确分解为 4 个子任务（pick soup → place → pick sauce → place），每轮之间 PERCEIVE，PRAE 闭环完整执行 |
| **LIBERO** | Franka Panda 在 LIBERO 仿真中依次将 2 个物体放入篮子，VLM 中间检查 + 最终评估确认全部成功 |

### 7.2 任务 B：链式型分解（Mock + LIBERO）

**LIBERO 任务**: `put the black bowl in the bottom drawer of the cabinet and close it` (KITCHEN_SCENE4)

**验证重点**: LLM 能识别动作间的依赖关系（先开抽屉才能放入，放入后才能关），分解为 4 个不同类型的顺序子任务。

| 模式 | 通过条件 |
|------|---------|
| **Mock** | LLM 正确分解为 4 个子任务（open drawer → pick bowl → place in drawer → close drawer），识别依赖顺序，PRAE 闭环完整执行 |
| **LIBERO** | Franka Panda 在 LIBERO 仿真中完成 开抽屉→抓碗→放入→关抽屉 全链路，VLM 在关键节点（抽屉开了？碗放进去了？）中间检查 + 最终评估确认 |

### 7.3 共性验收标准

- 两个任务 LLM 都能正确判断为 hard 难度并选择对应路由
- 每个子任务由 VLA 控制循环独立执行，LLM 在子任务间做调度决策
- 失败时自动重试（最多 3 次），3 次失败后合理报告当前状态和卡在哪一步
- E-Stop 工具可随时中断执行
- Mock 模式无需 GPU，LIBERO 模式仅需 VLA 推理服务的 GPU

---

## 8. Future Scope

以下功能在 MVP 之后的版本中实现：

| 功能 | 说明 |
|------|------|
| **Multi-Robot** | 多机器人通过 MessageBus 协调，角色分工（scout / manipulator / monitor） |
| **Voice Input** | 板载麦克风（本地 Whisper ASR）+ Web Voice（浏览器 Web Speech API） |
| **Input Routing** | 统一 Input Router 归一化所有输入渠道为 ChannelMessage，支持优先级和限流 |
| **LoRA Hot-Swap** | 运行时切换 VLA 的 LoRA 权重，per-subtask 特化（< 2s 切换） |
| **Model Lifecycle** | 独立进程管理模型加载/卸载/VRAM 分配/健康监控 |
| **OTA Updates** | 增量拉取新 adapter/model 版本，校验 + 注册 + 归档旧版本 |
| **Autonomous Ops** | 巡逻路线、任务自动领取、展示表演模式、空闲交互模式 |
| **Sim Fork-Compare** | 并行仿真多种策略，对比结果，提升胜者到主环境 |
| **Web UI** | 浏览器界面：实时仿真画面 + 指令输入 + 状态仪表盘 |
| **Multi-Channel** | 启用 nanobot 内置 channels（Telegram、Slack、飞书、Email 等） |
| **4-Layer Safety** | 安全区域围栏 + 审计日志（在 E-Stop + Velocity 基础上扩展） |
| **Dynamic Routing** | 多路由配置（cautious、fast_manipulation）、三级难度、per-route LoRA |
| **VLM Terminator** | VLM 判断子任务是否完成，用于复杂多步任务 |
| **Navigation** | 移动底座导航，需切换 RoboCasa（PandaMobile）或 Habitat 等支持导航的仿真 |
| **Real Hardware** | Unitree G1 + GR00T N1.6（ZMQ adapter） |

---

## 9. Glossary

| 术语 | 定义 |
|------|------|
| **VLA** | Vision-Language-Action model。输入图像+文本指令，输出机器人动作序列 |
| **VLM** | Vision-Language Model。输入图像+文本问题，输出文本描述 |
| **LLM** | Large Language Model。文本推理和规划 |
| **PRAE** | Prepare → Perceive → Reason → Act → Evaluate，agent 的核心执行循环 |
| **Intention** | 逻辑层：快速意图识别和任务框定 |
| **Cognition** | 逻辑层：场景理解 + 推理规划，输出结构化 plan |
| **Action** | 逻辑层：高频机器人控制循环（VLA + 安全 + 执行） |
| **Tool** | 注册在 agent 中的异步函数，LLM 可通过 tool_call 调用。机器人动作和软件操作是同一抽象 |
| **LoopManager** | 管理 VLA 控制循环的组件，处理启动/等待/终止/统计 |
| **Terminator** | 判断 VLA 控制循环何时应停止的策略 |
| **Action Horizon** | VLA 每次推理产出的动作步数（chunking size） |
| **Route** | YAML 配置，指定不同任务难度下的控制参数（频率、步数限制、终止策略） |
| **LIBERO** | 基于 MuJoCo 的机器人桌面操作仿真平台，Franka Panda 固定底座，130+ 任务（5 个 suite），7 维动作空间（3D 位移 + 3D 旋转 + 夹爪），SmolVLA/pi0/openpi 原生支持 |
| **Nanobot** | 超轻量 Python agent 框架（~4,000 LOC），本项目的基础 |
| **SKILL.md** | 技能描述文件，YAML frontmatter + Markdown 指令，agent 可加载并按步骤执行 |
