# 交接包验证记录

2026-09-27，本地 Windows/Python 环境执行，未占用 GPU。

```bash
python -B -m unittest discover -s tests -v
```

结果：20 项，其中 **19 通过、1 跳过**。跳过项为真实文件系统 symlink 逃逸测试，原因是 Windows 当前账户没有创建符号链接的权限；Linux 接手环境应重新运行。独立的根目录外输入拒绝测试已通过。

覆盖：

- 八模型结构候选数与签名，总计296；逐个约束、错误 world size、协议修改拒绝。
- 重复/缺失候选记录检测、模型声明 hash 变化、新输出目录与拒绝覆盖。
- 完整源码/输入缺失、两后端 tokenizer 路径、真实 cases8 schema。
- 公开源码采集与 private_inputs 分离，旧 Profile 排除，超大源码拒绝。
- 疑似密钥报告不泄露密钥值、根目录越界拒绝。

参考文件哈希由 source_manifest.json 记录；历史账号/节点名称替换为 LEGACY 占位符。扫描未发现需要发布的真实凭证；扫描并非任意内容安全性的数学保证。

未做：H20 软件安装、H20 CUDA kernel、NCCL 通信、真实后端参数对齐、显存可行性、训练、搜索性能。不能将本记录写成实验成功或论文数据。
