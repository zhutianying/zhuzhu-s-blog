const { copyFileSync, existsSync, mkdirSync } = require('node:fs');
const { dirname, resolve } = require('node:path');

const source = resolve(__dirname, '../node_modules/decap-cms/dist/decap-cms.js');
const destination = resolve(__dirname, '../source/admin/decap-cms.js');

if (!existsSync(source)) {
  throw new Error('Decap CMS is missing. Run npm ci before building the site.');
}

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
