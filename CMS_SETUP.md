# 配置在线写作后台

后台地址是 <https://zhutianying.com/admin/>。它使用 Decap CMS 编辑 `source/_posts` 中的文章，并经 GitHub OAuth 直接把变更提交到 `main`；GitHub 的提交会照常触发 Vercel 部署。

## 1. 创建 GitHub OAuth App

进入 [GitHub OAuth Apps](https://github.com/settings/developers)，选择 **New OAuth App**，填写：

| 字段 | 值 |
| --- | --- |
| Application name | `zhuzhu's blog CMS` |
| Homepage URL | `https://zhutianying.com` |
| Authorization callback URL | `https://zhutianying.com/api/callback` |

注册后记下 **Client ID**，并在该应用页面点击 **Generate a new client secret**。Client secret 只显示一次，不要提交、截图或发送给任何人。

## 2. 在 Vercel 配置环境变量

进入 Vercel 项目 → **Settings** → **Environment Variables**，添加以下三项到 Production、Preview 和 Development：

| 变量名 | 值 |
| --- | --- |
| `GITHUB_OAUTH_CLIENT_ID` | GitHub OAuth App 的 Client ID |
| `GITHUB_OAUTH_CLIENT_SECRET` | GitHub OAuth App 的 Client secret |
| `CMS_SITE_URL` | `https://zhutianying.com` |

添加变量后，执行一次 **Redeploy**，让 Function 读取新变量。

## 3. 登录和写作

访问 <https://zhutianying.com/admin/>，选择 **Login with GitHub**。只有对 `zhutianying/zhuzhu-s-blog` 有写权限的 GitHub 账号才能发布文章。

文章和上传图片会由 CMS 直接提交到 `main`；每次发布后等待 Vercel 构建完成即可在网站上看到更新。

## 安全说明

- GitHub OAuth Client secret 仅存在 Vercel 环境变量，绝不能写入仓库。
- OAuth 使用短期随机 `state`、PKCE 和 `HttpOnly`/`Secure` Cookie 保护授权回调。
- `/admin/` 页面可以公开访问，但未经 GitHub 登录、且没有该仓库写权限的用户无法修改内容。
- 本站仓库是公开仓库，因此 OAuth 只申请 `public_repo` 权限；它足以编辑这个公开博客。需要撤销时，可在 GitHub 的 Applications 设置中撤销对该 OAuth App 的授权。
