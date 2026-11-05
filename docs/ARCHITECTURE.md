# 项目结构

这是一个由 Vercel 部署的 Hexo 静态博客。`main` 分支的每次推送都会执行 `npm run build`，并发布 `public/` 生成物。

```text
source/_posts/       文章内容与 Front Matter
source/admin/        Decap CMS 的静态后台配置
themes/aircloud/     EJS 模板、样式和前端交互
api/                 Vercel Functions：CMS 的 GitHub OAuth 回调
scripts/             构建期辅助脚本
test/                Node 内置测试
```

## 内容约定

- 每篇文章必须包含 `title`、`date`、`categories`、`tags` 和 `description`。
- 全站时区固定为 `Asia/Shanghai`；未来时间的文章不会被生产构建发布。
- 文章图片优先放在文章同名的资源目录；CMS 上传图像位于 `source/images/uploads`。

## 本地验证

运行 `npm run check`。它会执行 OAuth 回归测试，并从零生成静态站点。发布前不要手动提交 `public/` 或 `source/admin/decap-cms.js`。

## 部署与边界

Vercel 负责 HTTPS、静态文件和 `api/` 中的函数。`CMS_SITE_URL` 必须是纯源地址，例如 `https://zhutianying.com`。OAuth 凭据只放在 Vercel 环境变量，不写入仓库。
