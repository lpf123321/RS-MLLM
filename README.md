# 多模态遥感大模型

## 环境配置

```bash
# 克隆仓库
git clone git@github.com:lpf123321/RS-MLLM.git
cd RS-MLLM

# 一键安装环境（自动建 conda 环境 + 装所有依赖）
bash setup.sh
```

## 模型权重

模型权重和数据集存放在共享区，需要手动建立软链接：

```bash
# 模型
mkdir -p models
ln -s /users/u2024311136/shared/shared_models/Qwen3-VL-2B-Instruct models/Qwen3-VL-2B-Instruct

# 数据集
mkdir -p datasets
ln -s /users/u2024311136/shared/shared_datasets datasets
```

## 数据预处理

### VRSBench（已完成预处理）

VRSBench 训练和评测数据已预处理为 Qwen3-VL 的 `messages` 格式，位于共享数据集目录中：

```
datasets/VRSBench/
├── vrsbench_train.jsonl    # 训练数据，142,390 条
└── vrsbench_eval.jsonl     # 评测数据，62,918 条
```

建立好 `datasets` 软链接后可直接读取，无需再运行预处理脚本。

**格式说明**（每条数据一个 JSON 对象，每行一条）：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/absolute/path/to/image.png"},
        {"type": "text", "text": "[VQA] 指令文本"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "答案文本"}
      ]
    }
  ]
}
```

**任务前缀说明**：

| 前缀 | 任务 | 示例输出 |
|------|------|---------|
| `[VQA]` | 视觉问答 | `"expressway-toll-station"` |
| `[CAP]` | 图像描述 | `"The image shows..."` |
| `[REF]` | 指代表达定位 | `"{<45><45><59><59>}"` |

**使用方式**：微调时直接读取 `datasets/VRSBench/vrsbench_train.jsonl` 和 `datasets/VRSBench/vrsbench_eval.jsonl` 即可。图像路径为绝对路径，需确保软链接 `datasets → /users/u2024311136/shared/shared_datasets` 已建立。

> 如需从原始标注重新生成，可运行 `python scripts/preprocess_vrsbench.py`。

## 提交代码

```bash
# 1. 拉取最新代码
git checkout main && git pull

# 2. 创建自己的分支
git checkout -b feat/你的功能名

# 3. 提交并推送
git add .
git commit -m "feat: 描述你的改动"
git push origin feat/你的功能名

# 4. 在 GitHub/GitLab 上创建 Pull Request
```
