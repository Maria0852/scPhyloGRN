# scPhyloGRN 中文上手指南：从打开 notebook 到保存预测

主教程是 `quickstart.ipynb`。它用真实训练的 decoder 权重和预先计算的基因特征在 CPU 上预测，不训练、不重建大图。英文教程里的每个代码格只完成一个小任务，操作说明放在前面的文字格。

## 1. 第一次应该打开哪个文件？

- 打开 `quickstart.ipynb`，不要先运行需要 GPU 的 `inference_demo.ipynb`。
- 如果拿到 ZIP，请先完整解压；不要只拷贝 `.ipynb`。辅助代码、权重和缓存必须保留在同一目录结构中。
- 输入是两列基因名，输出是每个有向基因对的模型分数。这里不能上传新的表达矩阵或新基因。

## 2. 在当前集群上怎样运行？

交互方式：在 JupyterHub 中申请 **CPU 计算节点、2 个 CPU 核、2 GB 内存**，然后打开 notebook，选择安装了所需依赖的 Python 内核。不要在登录节点启动 Jupyter、安装依赖或执行 Python。终端中申请了节点，并不会把已经打开的 notebook 内核搬过去。

先运行第一个代码格，查看主机名和 Slurm 作业编号。如果提示没有申请节点，停止并重新启动计算节点上的 Jupyter 会话，不要删除检查或手动伪造环境变量。

无需交互的方式（完整仓库内）：

```bash
cd /export/home/mengrui/mycode/gnn_grn/zenodo_release
sbatch run_quickstart.sbatch
```

这条命令只提交任务；打包、推理、绘图和验证全部在申请节点中运行。脚本会把当前教程重新打包，然后在仓库外解压执行，避免测试旧 ZIP。执行结果见 `artifacts/quickstart.executed.ipynb`，日志见 `artifacts/quickstart-作业编号.out` 和 `.err`。批处理中的结果目录位于临时解压目录，具体位置见日志及已执行 notebook 的最后一个格子；交互运行则保存在当前解压目录的 `results/` 内。

本集群脚本默认复用现有 Conda 环境。若要指定其他环境，提交时设置 `PYTHON_BIN` 和 `NOTEBOOK_PYTHON`；二者必须分别包含推理依赖和 notebook 执行工具。

## 3. 在个人电脑上怎样运行？

无需 Slurm 和 GPU。建议从 Python 3.10 的独立虚拟环境开始。打开终端，进入 ZIP 解压目录，在那里执行：

```bash
python -m venv .venv
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv/Scripts/Activate.ps1
```

然后安装并启动：

```bash
python -m pip install -r requirements-quickstart.txt
python -m jupyter lab
```

浏览器打开后选择 `quickstart.ipynb`，确认内核属于刚安装的环境。个人电脑的全新安装与集群现有环境不是同一个测试场景，实际验证范围以执行记录为准。

## 4. 每个格子怎么运行？

1. 第一次不要修改示例，从上到下运行。
2. 文字格负责解释；带 `[ ]` 的代码格负责执行。选中代码格后按 **Shift + Enter**。
3. `[*]` 表示正在运行，变成数字后再继续。
4. 如果内核重启或出现 `NameError`，从头执行，不要直接跳到预测格。
5. 路径错误时，在第 1.1 节把 `BUNDLE_DIR` 改为完整解压目录。

## 5. 你会看到什么？

- 第 1 节：简短文字说明，区分预先完成的 encoder 与本次运行的 decoder。
- 第 2 节：模型身份、基因数量、可用基因名示例。
- 第 3 节：五行输入和箭头方向图。箭头是“要问的问题”，不是已验证的调控关系。
- 第 4–5 节：真实预测表和分数图；左侧是原始 logit，右侧是 0–1 范围的校准分数。
- 第 6 节：示例校验通过的 PASS 提示。
- 第 7–8 节：自己的查询、图，以及保存后重新读取 CSV 的检查。

## 6. 怎样换成自己的输入？

第一次跑通后，只改第 7.1 节 `my_pairs` 中引号里的基因名；也可以在第 7.2 节指定自己的 CSV。CSV 必须有 `gene_i,gene_j` 两列，基因名必须出现在 `model/quickstart/genes.csv` 中，并区分大小写。

只改基因对时，重跑第 7–8 节即可。不要改前面的示例或参考结果。A→B 与 B→A 是不同查询；重复有向对、自身到自身、空值和未知基因会被明确拒绝。

## 7. 结果在哪里？

第 8 节为每次运行创建 `results/时间戳/`，保存两份 CSV、三张 PNG 和一份 PDF。最后的格子打印具体位置。JupyterLab 左侧进入对应目录，右键文件选择 Download 即可下载。

模型分数不是实验确认的调控概率。示例可能与训练数据重叠，图也不表示独立测试性能。快速入门 ZIP 不含完整 encoder 和原始表达数据，不能替代完整模型/数据发布；当前公开 DOI 和再分发许可仍待确认。
