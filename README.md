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
