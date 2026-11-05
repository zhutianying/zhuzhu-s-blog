# zhuzhu's blog

[zhutianying.com](https://zhutianying.com/) 的源码。它是一个 Hexo 静态站点：文章和图片存放在 GitHub，推送到 `main` 后由 Vercel 自动构建并发布，因此不需要云服务器。

## 日常写作

Node.js 22 是此项目的唯一受支持版本。安装依赖后，用下面的流程写作：

```bash
npm ci
npm run new -- "我的文章标题"
npm run dev
```

新建文章在 `source/_posts/`；本地预览地址为 <http://localhost:4000>。文章写好后，提交并推送到 `main`：

```bash
git add source/_posts
git commit -m "post: 我的文章标题"
git push origin main
```

Vercel 会运行 `npm run build` 并发布 `public/` 目录。`public/` 是生成物，不应手动修改或提交。

## 文章格式

新文章会自动使用 `scaffolds/post.md` 模板。至少填写标题、分类和标签：

```md
---
title: 一个可复用的工作方法
date: 2026-11-05 20:00:00
categories:
  - AI 与开发
tags:
  - Hexo
  - 工作流
description: 用一句话说明文章要解决的问题。
---
```

图片请放进和文章同名的资源目录，例如：

```text
source/_posts/我的文章.md
source/_posts/我的文章/cover.png
```

再用 `![封面](cover.png)` 引用。项目已开启 Hexo 的文章资源目录功能。

## 本地检查

在推送前运行：

```bash
npm run check
```

该命令会先验证 OAuth 配置逻辑，再生成静态站点到 `public/`。不要用 `npm audit fix --force` 批量升级旧依赖；升级 Hexo 或主题前，应在单独分支验证页面样式和文章链接。

## Vercel 设置

在 Vercel Project Settings → Build and Deployment 中将 Node.js Version 设为 **22.x**。仓库中的 `engines.node`、`.nvmrc` 与 `vercel.json` 也会把本地和线上构建统一到同一套约定。

## 关于后台写作

线上后台已使用 Decap CMS 接入 GitHub OAuth，入口是 <https://zhutianying.com/admin/>。首次启用需要按 [CMS_SETUP.md](CMS_SETUP.md) 创建 GitHub OAuth App 并在 Vercel 设置三个环境变量；其中 `CMS_SITE_URL` 必须是纯站点源地址（例如 `https://zhutianying.com`），不能包含路径、查询参数或片段。凭据不存放在仓库中。

`hexo-admin` 仍是一个只适合**本地**使用的 Hexo 编辑界面，不是线上后台。线上后台发布文章时会直接提交 GitHub 的 `main` 分支，并自动触发 Vercel 部署。
