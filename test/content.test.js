const assert = require('node:assert/strict');
const { readdirSync, readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');

const postsDirectory = join(__dirname, '../source/_posts');
const requiredFields = ['title', 'date', 'categories', 'tags', 'description'];

function frontMatter(source) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  assert.ok(match, '文章必须以 YAML Front Matter 开头');

  return new Set(
    [...match[1].matchAll(/^([A-Za-z][A-Za-z0-9_-]*):/gm)].map(([, field]) => field),
  );
}

test('all posts provide the required SEO metadata', () => {
  const postFiles = readdirSync(postsDirectory).filter((file) => file.endsWith('.md'));
  assert.ok(postFiles.length > 0, '至少应有一篇文章');

  postFiles.forEach((file) => {
    const fields = frontMatter(readFileSync(join(postsDirectory, file), 'utf8'));
    requiredFields.forEach((field) => {
      assert.ok(fields.has(field), `${file} 缺少 ${field}`);
    });
  });
});
