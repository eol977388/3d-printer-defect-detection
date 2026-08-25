# GitHub修改内容上传与合并SOP

## 一、目的

将本地已经完成并审核的代码、配置、报告和SOP安全上传到GitHub，通过独立分支和Pull Request合并到`main`，形成可追溯、可恢复、可审查的项目版本。

本SOP适用于Windows CMD或PyCharm终端中的日常项目提交。

## 二、核心思想

1. **先检查，后暂存**：上传前先确认当前分支和文件状态。
2. **使用独立功能分支**：数据分析、训练配置、模型改进分别提交，避免直接修改`main`。
3. **明确选择上传文件**：优先使用`git add 指定目录`，不要习惯性执行`git add .`。
4. **数据与代码分开管理**：普通Git仓库保存代码、配置、报告和小型图表；不直接保存大型数据集、训练权重和运行目录。
5. **提交前复核暂存区**：确认没有误传隐私文件、数据集、权重或无关换行符修改。
6. **通过Pull Request合并**：在GitHub检查差异、冲突和自动化结果后再进入`main`。
7. **合并后同步本地main**：保证本地与远程主分支一致，再开始下一项工作。

## 三、上传范围

通常应上传：

- 源代码和数据处理脚本；
- 数据配置、模型配置和超参数文件；
- Markdown报告、SOP、CSV处理清单；
- 体积合理且用于报告的PNG图表；
- JSON实验或数据处理汇总；
- 环境及训练命令记录。

通常不上传：

- 原始数据集和清洗后的完整数据集；
- 人工审核产生的大批图片；
- `runs/`训练输出；
- `.pt`、`.weights`等权重；
- 缓存、临时文件、IDE配置和账号密钥。

如确需管理大文件，应单独评估Git LFS、DVC、对象存储或数据平台，不能直接使用普通Git提交。

## 四、标准步骤

### 步骤1：进入项目目录

```bat
cd /d D:\ruanjianxiazai\pycharm2026\yolov5-7.0
```

注释：`/d`允许CMD同时切换盘符和目录。

### 步骤2：检查当前状态

```bat
git status
```

注释：确认当前分支、未跟踪文件、已修改文件和已暂存文件。

```bat
git branch --show-current
```

注释：只显示当前分支名称，避免把功能内容直接提交到错误分支。

### 步骤3：创建或切换功能分支

首次创建分支：

```bat
git switch -c data-analysis
```

注释：从当前提交创建并切换到`data-analysis`分支。

分支已存在时：

```bat
git switch data-analysis
```

注释：切换到已有功能分支，不重复创建。

分支名称应体现任务，例如：

```text
data-analysis
training-config
baseline-training
model-improvement
```

### 步骤4：明确暂存本次文件

本项目数据处理阶段采用：

```bat
git add tools reports
```

注释：只暂存数据处理脚本和报告，不把其他无关修改加入提交。

不建议直接执行：

```bat
git add .
```

原因：可能误加入数据集、临时文件、权重、换行符变化或不属于本任务的修改。

### 步骤5：处理被`.gitignore`忽略但确实需要的报告文件

YOLOv5原始`.gitignore`可能全局忽略PNG和JSON。如果报告必须包含图表和汇总，可对明确目录强制添加：

```bat
git add -f reports
```

注释：`-f`强制添加`reports`中被忽略的报告图和JSON；使用前必须确认该目录没有数据集或敏感文件。

出现以下提示通常不是错误：

```text
LF will be replaced by CRLF
```

含义：Windows Git可能调整工作区换行符，不代表JSON或代码损坏。

### 步骤6：复核暂存区

```bat
git status
```

注释：检查`Changes to be committed`，确认文件范围正确。

```bat
git diff --cached --stat
```

注释：查看待提交文件数量和总体变更规模。

```bat
git diff --cached
```

注释：查看文本文件的具体改动；较大提交至少检查关键配置、脚本和报告。

中文文件名显示为转义字符时，可执行：

```bat
git config core.quotepath false
```

注释：让`git status`正常显示中文路径，不改变文件内容。

### 步骤7：创建本地提交

```bat
git commit -m "Add reproducible dataset cleaning and clean_v2 analysis"
```

注释：将暂存内容保存为一个本地提交；提交说明应概括完成的结果。

推荐提交信息格式：

```text
Add ...
Fix ...
Update ...
Document ...
```

一个提交尽量只对应一个明确任务。

### 步骤8：上传功能分支

```bat
git push -u origin data-analysis
```

注释：首次上传本地分支，并建立对远程同名分支的跟踪关系。

以后继续更新该分支时可直接执行：

```bat
git push
```

注释：把新增提交推送到已经关联的远程分支。

### 步骤9：创建Pull Request

在GitHub进入`Compare & pull request`或使用终端提供的链接，确认：

```text
base: main
compare: data-analysis
```

标题应清晰描述结果，例如：

```text
添加可复现的数据集清洗代码，以及clean_v2版本的分析
```

说明应包含：

- 主要修改；
- 核心数据或实验结果；
- 数据处理原则；
- 已上传和未上传的内容；
- 验证方式及已知限制。

然后点击`Create pull request`。

### 步骤10：检查并合并Pull Request

合并前确认：

- 目标分支是`main`；
- Files changed符合预期；
- 页面显示无分支冲突；
- 失败检查是否与本次代码质量相关。

本项目出现`Greetings / greeting`失败。它是欢迎PR的自动化流程，不是代码或数据检查；页面同时显示无冲突且允许合并，因此不阻止本次合并。

当功能分支只有一个完整提交时，推荐选择：

```text
Squash and merge
```

注释：将功能分支内容合并为main中的一个清晰提交。

操作顺序：

1. 从合并按钮下拉菜单选择`Squash and merge`；
2. 点击绿色`Squash and merge`；
3. 点击`Confirm squash and merge`。

### 步骤11：同步本地main

```bat
git status
```

注释：切换分支前确认没有未提交工作。

```bat
git switch main
```

注释：切换回本地主分支。

```bat
git pull origin main
```

注释：拉取GitHub中刚合并的最新主分支。

```bat
git status
```

注释：正常应显示本地`main`与`origin/main`一致且工作区干净。

### 步骤12：可选清理功能分支

确认合并和本地同步均正常后：

```bat
git branch -d data-analysis
```

注释：删除已经合并的本地功能分支，不影响`main`中的内容。

如果GitHub页面仍保留远程分支，可点击`Delete branch`，或执行：

```bat
git push origin --delete data-analysis
```

注释：删除远程功能分支；Pull Request和main中的历史仍然保留。

## 五、异常处理

### 1. `fatal: not a git repository`

原因：当前目录没有`.git`。先进入正确项目目录；新项目才执行`git init`。

### 2. `nothing added to commit but untracked files present`

原因：文件存在，但没有执行`git add`。明确选择目录后再暂存。

### 3. Pull Request存在冲突

先更新功能分支并解决冲突，不应在不了解冲突内容时强制合并。

### 4. 自动检查失败

区分代码测试、格式检查、安全检查和非关键欢迎流程。关键测试失败应先修复；与代码无关且不阻止合并的流程应记录原因后处理。

### 5. GitHub拒绝大文件

普通GitHub单文件上限通常为100MB。不要尝试反复提交，应从暂存区移除，并改用Git LFS、DVC或外部存储。

## 六、本次操作结果

- 功能分支：`data-analysis`
- Pull Request：`#16`
- 合并方式：`Squash and merge`
- 合并目标：`main`
- main合并后提交：`7dce7f5`
- 已上传：分析和清洗脚本、raw_v1及clean_v2报告、图表、清洗记录、SOP
- 未上传：原始数据、clean_v1、clean_v2、人工审核大图、训练权重、训练结果

本次实现了“本地修改—明确暂存—功能分支推送—Pull Request审核—Squash合并—同步main”的完整可复现流程。

## 七、提交前最终检查清单

- [ ] 当前分支正确
- [ ] 工作区修改属于本任务
- [ ] 暂存区不含数据集、权重、密钥和临时文件
- [ ] 报告引用的必要图表和JSON已加入
- [ ] 提交信息能描述结果
- [ ] PR的base与compare正确
- [ ] Files changed已检查
- [ ] 关键自动检查通过或失败原因已确认
- [ ] PR已合并
- [ ] 本地main已同步
