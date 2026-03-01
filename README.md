# 5-Kronos-Predictor-GUI 🔮🌌

**KLine Kronos Suite - 脑机推演终端 (Program 5)**

这是整个矩阵套件的最终收口与应用落地层。作为您的私有本地化 Kronos Demo 测试台，它被设计用于将历史真实行情流输入已觉醒的神经网络中，抽取平行宇宙的未来分支。

## 🌟 核心特性 (Key Features)

- **完全解耦与挂载**: 此程序自身不包含训练逻辑。它可以自由挂载 `4-Kronos-Trainer-Base/models/` 目录下您微调出来的专属 `my_finetuned_tokenizer` 和 `my_finetuned_predictor` 大脑。
- **特征对齐与缩放**: 内置完整的推演前置处理链 (Preprocessing Pipeline)。读取最新的 CSV 后，即时进行特征维度检查（映射 vol->volume 等）、执行截面 `Transform` 归一化。
- **高维空间降维**: 当神经网络在潜在空间中预测出未知的走势 Token 时，脚本会再次执行 `Inverse Transform`，将模型输出转译回“人类可读”的真实股价刻度序列（如 24.52 元）。
- **参数控制面板**: 可以在界面上调节生成引擎的“想象力 (Temperature)”、“Lookback 窗口 (默认 90)” 及“预测步长 (默认 10)”。
- **全息预测导出**: 推演完成后，自动压制一份带有时序标记的 `_prediction.csv` 将其写入特定的 `predictions/` 收容区。

## 🛠 使用流线

1. 确认已在 Program 4 中成功炼成了专属大脑模型。
2. 启动终端。选取一个近期的股票 CSV (建议为 5 分钟级)。
3. 点击 **⚡ 启动脑机推演序列**。
4. 程序将一键完成推理并导出未来 10 个数据点的绝对价格供图表比对。

---
`Design Language: Cyber-Gold (Flat Dark Gold)`
