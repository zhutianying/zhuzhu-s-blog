const assert = require('node:assert/strict');
const { existsSync, readFileSync, readdirSync } = require('node:fs');
const { resolve } = require('node:path');

const publicDirectory = resolve(__dirname, '../public');
const read = (path) => readFileSync(resolve(publicDirectory, path), 'utf8');
const latestTitle = '2026年11月05日早间全球资讯简报';

const home = read('index.html');
assert.match(home, new RegExp(latestTitle));
assert.match(home, /<div class="post-content/);

const archive = read('archive/index.html');
assert.match(archive, new RegExp(latestTitle));
assert.match(archive, /post-preview-container/);

const aiTag = read('tags/AI/index.html');
assert.match(aiTag, /ChatGPT的技术逻辑及演进历程/);

const morningNewsTag = read('tags/早间新闻/index.html');
assert.match(morningNewsTag, new RegExp(latestTitle));

assert.match(home, /友链：/);
assert.match(home, /GodKai/);
assert.match(home, /拾梦人/);
assert.match(home, /京ICP备2023003608号-1/);
assert.doesNotMatch(home, /Created By/);
assert.match(home, /<strong>股市。<\/strong> 11 月 4 日/);
assert.doesNotMatch(home, /\*\*/);

const postsDirectory = resolve(__dirname, '../source/_posts');
for (const file of readdirSync(postsDirectory).filter((name) => name.endsWith('.md'))) {
  const source = readFileSync(resolve(postsDirectory, file), 'utf8');
  assert.match(source, /^tags: [^\n\[\]-]+$/m, `${file} 必须只有一个标签`);
}

assert.equal(existsSync(resolve(publicDirectory, 'search.json')), false, '不应生成搜索索引');
assert.equal(existsSync(resolve(publicDirectory, 'archive/2024/index.html')), false, '不应生成旧归档页');
assert.equal(existsSync(resolve(publicDirectory, 'about/index.html')), false, '不应生成关于页面');
assert.equal(existsSync(resolve(publicDirectory, 'collect/index.html')), false, '不应生成收藏页面');
assert.equal(existsSync(resolve(publicDirectory, 'tags/金融/index.html')), false, '不应生成金融标签页');
assert.equal(existsSync(resolve(publicDirectory, 'tags/GitHub/index.html')), false, '不应生成 GitHub 标签页');
assert.equal(existsSync(resolve(publicDirectory, 'tags/国际时政/index.html')), false, '不应生成国际时政标签页');
