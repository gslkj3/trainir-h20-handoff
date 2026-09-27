# GitHub 下载、推送与可选 30 天凭证

目标仓库：`gslkj3/trainir-h20-handoff`。用户最终选择 Public，并允许不用令牌。**本次不创建密钥。** 下载公开内容无需认证；推送修改仍需被授权的 GitHub 身份。本文件不包含任何密钥。

## 最小权限

仅在后续需要独立写入凭证时，用户可创建 **fine-grained personal access token**，名称建议 `H20-TrainIR-30d`：

- Resource owner：`gslkj3`。
- Expiration：**30 days**（从实际生成日期起，不是写在文档里便自动生效）。
- Repository access：**Only select repositories**，只选 `trainir-h20-handoff`。
- Repository permissions：Contents **Read and write**；Metadata **Read-only**。
- 不授予 Administration、Workflows、Secrets、其他仓库或账户权限。

[打开预填名称、30天期限和 Contents 读写权限的 GitHub 创建页](https://github.com/settings/personal-access-tokens/new?name=H20-TrainIR-30d&description=H20%20TrainIR%20handoff%20only&target_name=gslkj3&expires_in=30&contents=write)

链接不能代替选择仓库，必须在网页检查 Only select repositories。由用户在 GitHub 页面完成创建并立即保存到安全位置，不将令牌粘贴到聊天或仓库。

GitHub 插件当前能够上传内容，但未开放创建 PAT 的接口。随机字符串不是 GitHub 令牌；普通 SSH deploy key 也没有等价的原生30天自动到期设置，不能用它假装完成期限要求。

## 在 H20 下载

公开仓库可以直接匿名下载：

```bash
git clone https://github.com/gslkj3/trainir-h20-handoff.git
```

只有需要推送时才认证：按终端提示填写 GitHub 用户名，Password 位置由用户输入令牌。不要把令牌放进 URL、命令参数、截图、shell history、git config remote 或实验日志。若用 GitHub CLI，则使用其交互认证及安全凭据存储；不安装来路不明的 credential helper。没有写入凭据时可在 H20 本地保存补丁和摘要，由用户带回上传，不能认为公仓允许匿名推送。

进入仓库后，agent 创建 `h20/` 前缀的新分支并推送修复与经脱敏的摘要。此权限是仓库范围，不是分支级隔离；不要将别的项目放入同一凭证授权范围。

到期后 GitHub 拒绝继续认证，代码副本仍会留在 H20；令牌到期不会撤回已下载数据。任务提前结束或服务器交还前，用户应主动撤销令牌并按自己的数据留存规则处理副本。

来源：[GitHub 官方个人访问令牌说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)。
