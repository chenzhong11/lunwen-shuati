# 论文刷题｜GitHub 同步约定

GitHub 仓库：<https://github.com/chenzhong11/lunwen-shuati>

本仓库是论文复现与机理优先学习的公开事实层，主要保存：

- 可复现代码、配置和测试；
- `01_notes/` 中的学习报告、阶段报告、审计记录和方法说明；
- `04_experiments/` 中可复用的正式图表和实验过程；
- `02_code/` 中的代码、配置和测试。

## 不上传的内容

- `03_data/`：原始数据和处理后的数据文件；
- `04_experiments/_codex_workspace/`：临时脚本、中间文件和调试产物；
- Python 缓存、IDE 配置、训练日志和可重新生成的实验输出；
- 密钥、令牌、个人隐私和本地路径中不适合公开的信息。

## 每轮产出后的同步方式

1. 将正式 Markdown 放入 `01_notes/`。
2. 将正式图表放入 `04_experiments/`。
3. 运行脚本和必要检查，确认不包含数据文件或临时文件。
4. 查看 `git status`，只提交本轮正式产出。
5. 使用清晰的阶段性提交信息推送到 `main`。

Notion 用于保存项目上下文、决策理由和知识沉淀；GitHub 用于保存代码、正式文件、版本历史和可复现证据。两者通过项目名称、报告日期和提交记录相互对应。

## 文件命名规则

- 面向使用者的正式 Markdown 统一使用英文小写 `lower_snake_case.md`。
- 阶段学习材料使用 `dayN_topic[_version].md`，例如 `day4_cnn_from_cwt.md`。
- 临时脚本、中间文件和调试记录只放在 `04_experiments/_codex_workspace/` 或项目外归档目录，不与正式资料混放。
