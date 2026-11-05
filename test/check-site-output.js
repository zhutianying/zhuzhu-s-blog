const assert = require('node:assert/strict');
const { existsSync, readFileSync } = require('node:fs');
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
assert.match(aiTag, new RegExp(latestTitle));
assert.match(aiTag, /ChatGPT的技术逻辑及演进历程/);

assert.equal(existsSync(resolve(publicDirectory, 'search.json')), false, '不应生成搜索索引');
assert.equal(existsSync(resolve(publicDirectory, 'archive/2024/index.html')), false, '不应生成旧归档页');
assert.equal(existsSync(resolve(publicDirectory, 'about/index.html')), false, '不应生成关于页面');
assert.equal(existsSync(resolve(publicDirectory, 'collect/index.html')), false, '不应生成收藏页面');
