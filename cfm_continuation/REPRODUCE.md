# 复现与恢复

从项目根目录执行。原数据位于 `cat_binary_dataset/`，Python 为 `cfm_v0/.venv/Scripts/python.exe`；依赖版本见 `cfm_v0/requirements.txt`。本轮没有升级环境。源数据、旧checkpoint、新checkpoint的SHA256分别见 `original_identity.json` 和 `checkpoint_inventory.json`。

新环境第一次运行（不能在已有run中重复prepare/configure）：

```powershell
& .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\prepare.py
foreach ($batchSize in 16,32,64) {
  & .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\training.py benchmark --batch $batchSize
}
& .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\training.py configure
& .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\check_resume.py
& .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\workflow.py milestone --step 20000
```

实际查看全部development图片后记录对应 `reviews/020000.json`，再用 `record_review.py --step 20000` 将观察与pixel hashes绑定。此脚本只记录人工提供的观察文本，不会自动看图。然后：

```powershell
& .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\workflow.py train --step 40000
& .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\workflow.py milestone --step 40000
& .\cfm_v0\.venv\Scripts\python.exe cfm_continuation\status.py --step 40000
```

训练每5k保存完整 `run/checkpoints/last.pt`。后续 `train --step 60000` 等自动从此状态恢复，`--step` 是本次milestone边界，不修改冻结的整体LR/safety schedule。训练根据固定预算停止时，实际step以 `progress.json` 为准，不能假定达到传入的milestone。每20k重复同targets/seeds生成、FEM和实际图片检查。累计80k后才可按冻结规则判断平台停止；如触发规定条件并决定额外restart，执行 `restart.py --reason "已记录理由"`，该工具仅修改schedule metadata、保留所有训练状态，并禁止第二次额外restart。

停止后运行 `finalize.py --reason "实际停止理由"` 提交停止与best-validation选择，再检查所选checkpoint的development图。如果best不在20k milestone，先按其实际step执行一次development评估与记录。随后 `workflow.py final` 运行640张历史final panel、真实FEM、指标和NN诊断。不得在final结果后继续调参或重新选择checkpoint。实际执行的所有命令与退出状态见 `run/commands.jsonl`，早期setup命令见 `run/setup_commands.json`。

本轮实际执行中，用户在120k评估后认可0.5182%的近期改善已足够接近平台，要求据此结束长训。运行中的140k segment改在125k原子完整checkpoint保存后由`stop_after_checkpoint.py`受控停止，保留完整恢复状态。对应parent workflow非零退出是预期的停止记录；不能解释成数值发散或文件损坏。该脚本包含此次task-owned进程身份检查，PID不是可复用的训练配置。复现学习轨迹时直接将该segment边界设为`train --step 125000`即可；若要继续训练，`last.pt`保留了125k的全部状态，但不得将继续后的结果混入本轮已经封存的final评估。

本轮实际finalize使用`--checkpoint last`和已记录的`--selection-reason`：125k development视觉/FEM优于120k，validation仅相对高约0.074%。`run/finalization_command.json`保留实际完整命令。`best-validation.pt`为120k真正的validation optimum，`last.pt`为125k完整状态；`best.pt`仅是为复用旧final sampler所保留的所选125k EMA别名。最小validation选择仍是默认规则，这次根据预先允许的视觉优先override选择last，且没有查看final结果后改选。

外部AI不需要weights即可复核存储数值：

```powershell
python cfm_continuation/verify_results.py --run cfm_continuation/run
```

重算已提供的binary outputs的FEM不需要训练图或模型权重。为避免读取现有FEM缓存，复制某个panel的 `binary.npy` 和 `manifest.json` 到新目录，例如 `cfm_continuation/run/fem_recheck/`，再执行：

```powershell
$env:CFM_V0_RUN_DIR = (Resolve-Path .\cfm_continuation\run).Path
python cfm_v0/fem.py evaluate --folder fem_recheck --workers 8
```

这会保持原FEM solver与material labels。模型完整重跑则需要另外取得ZIP中明确省略的原始training images和checkpoint。Gaussian sampling、BF16及cuDNN autotuning不保证跨软件/硬件bitwise相同。
