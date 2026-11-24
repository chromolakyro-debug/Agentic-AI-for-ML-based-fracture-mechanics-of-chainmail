# CFM v1 continuation 1 审阅入口

先读REPORT.md、training_curves.png和run/comparison_with_parent.json。final_evaluation/index.html显示最终240张生成图及两组train-only机械retrieval。run/fixed_seed_*.png是固定seed730000、全部24个targets的未经筛选对照，visual_observation文件记录实际人工查看。

run/training.jsonl前280k为父v1原始历史，其后为本次续训；完整状态继承，只有停止门槛和输出/保留规则改变。run/config.json、parent_config.json、training_code_changes.diff说明差异；checks.json为恢复验证，verification.json与final_retrieval_html_audit.json为最终核验。run/periodic_numerical_results和parent_final_results.json支持独立核算；final_fem_inputs.npz是最终FEM二值输入。

不包含大模型权重和训练图片，checkpoint_inventory.json列保存路径和SHA256。README.md和run/commands.jsonl记录恢复/复现命令。此panel复用此前targets/seeds，不是新的holdout；不据此宣称消除memorization。
