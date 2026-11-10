const assert = require('node:assert/strict');
const { existsSync, readFileSync, readdirSync } = require('node:fs');
const { resolve } = require('node:path');

const publicDirectory = resolve(__dirname, '../public');
const read = (path) => readFileSync(resolve(publicDirectory, path), 'utf8');
const literalPattern = (value) => new RegExp(value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
const postsDirectory = resolve(__dirname, '../source/_posts');
const posts = readdirSync(postsDirectory)
  .filter((name) => name.endsWith('.md'))
  .map((file) => {
    const source = readFileSync(resolve(postsDirectory, file), 'utf8');
    const title = source.match(/^title: (.+)$/m)?.[1];
    const date = source.match(/^date: (.+)$/m)?.[1];
    assert.ok(title && date, `${file} 必须包含标题和发布日期`);
    return { file, source, title, date };
  });
const latestTitle = posts.reduce((latest, post) => (post.date > latest.date ? post : latest)).title;
const knownMorningNewsTitle = '早间新闻-2026年11月05日';

const home = read('index.html');
assert.match(home, literalPattern(latestTitle));
assert.match(home, /<div class="post-content/);

const archive = read('archive/index.html');
assert.match(archive, literalPattern(latestTitle));
assert.match(archive, /post-preview-container/);

const aiTag = read('tags/AI/index.html');
assert.match(aiTag, /ChatGPT的技术逻辑及演进历程/);

const morningNewsTag = read('tags/早间新闻/index.html');
assert.match(morningNewsTag, literalPattern(knownMorningNewsTitle));

assert.match(home, /友链：/);
assert.match(home, /GodKai/);
assert.match(home, /拾梦人/);
assert.match(home, /京ICP备2023003608号-1/);
assert.doesNotMatch(home, /Created By/);
const earlyMorningNews = read('2026/11/05/2026-11-05-早间新闻/index.html');
assert.match(earlyMorningNews, /<strong>A股｜11月4日收盘。<\/strong> 上证综指/);
assert.doesNotMatch(earlyMorningNews, /\*\*/);
assert.match(earlyMorningNews, /发布于：<span>2026-11-05 06:30:00<\/span>/);
assert.match(earlyMorningNews, /资讯截点：北京时间 2026-11-05 06:30/);
assert.equal(existsSync(resolve(publicDirectory, '2026/11/05/2026-11-05-早间新闻/index.html')), true, '早间新闻链接应保持稳定');

for (const post of posts) {
  assert.match(post.source, /^tags: [^\n\[\]-]+$/m, `${post.file} 必须只有一个标签`);
}

const cmsConfig = readFileSync(resolve(__dirname, '../source/admin/config.yml'), 'utf8');
assert.doesNotMatch(cmsConfig, /name: categories/, '后台不应显示分类字段');
assert.match(cmsConfig, /name: description, widget: hidden/, '后台应隐藏摘要字段');

assert.equal(existsSync(resolve(publicDirectory, 'search.json')), false, '不应生成搜索索引');
assert.equal(existsSync(resolve(publicDirectory, 'archive/2024/index.html')), false, '不应生成旧归档页');
assert.equal(existsSync(resolve(publicDirectory, 'about/index.html')), false, '不应生成关于页面');
assert.equal(existsSync(resolve(publicDirectory, 'collect/index.html')), false, '不应生成收藏页面');
assert.equal(existsSync(resolve(publicDirectory, 'tags/金融/index.html')), false, '不应生成金融标签页');
assert.equal(existsSync(resolve(publicDirectory, 'tags/GitHub/index.html')), false, '不应生成 GitHub 标签页');
assert.equal(existsSync(resolve(publicDirectory, 'tags/国际时政/index.html')), false, '不应生成国际时政标签页');
