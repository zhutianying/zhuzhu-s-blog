const assert = require('node:assert/strict');
const test = require('node:test');

const auth = require('../api/auth');
const callback = require('../api/callback');

function response() {
  return {
    headers: {},
    statusCode: 200,
    body: undefined,
    setHeader(name, value) {
      this.headers[name] = value;
    },
    status(code) {
      this.statusCode = code;
      return this;
    },
    send(body) {
      this.body = body;
      return this;
    },
    redirect(code, url) {
      this.statusCode = code;
      this.headers.Location = url;
      return this;
    },
  };
}

test('OAuth authorization uses the configured CMS origin instead of the request host', () => {
  const previousOrigin = process.env.CMS_SITE_URL;
  const previousClientId = process.env.GITHUB_OAUTH_CLIENT_ID;
  process.env.CMS_SITE_URL = 'https://zhutianying.com';
  process.env.GITHUB_OAUTH_CLIENT_ID = 'test-client-id';

  try {
    const res = response();
    auth({ method: 'GET', headers: { host: 'untrusted.example' } }, res);

    assert.equal(res.statusCode, 302);
    const redirect = new URL(res.headers.Location);
    assert.equal(redirect.origin, 'https://github.com');
    assert.equal(redirect.searchParams.get('redirect_uri'), 'https://zhutianying.com/api/callback');
    assert.equal(res.headers['Cache-Control'], 'no-store');
  } finally {
    if (previousOrigin === undefined) delete process.env.CMS_SITE_URL;
    else process.env.CMS_SITE_URL = previousOrigin;
    if (previousClientId === undefined) delete process.env.GITHUB_OAUTH_CLIENT_ID;
    else process.env.GITHUB_OAUTH_CLIENT_ID = previousClientId;
  }
});

test('OAuth endpoints reject a CMS site URL containing a path', async () => {
  const previousOrigin = process.env.CMS_SITE_URL;
  process.env.CMS_SITE_URL = 'https://zhutianying.com/admin';

  try {
    const authResponse = response();
    auth({ method: 'GET', headers: {} }, authResponse);
    assert.equal(authResponse.statusCode, 500);

    const callbackResponse = response();
    await callback({ method: 'GET', query: {}, cookies: {} }, callbackResponse);
    assert.equal(callbackResponse.statusCode, 500);
    assert.equal(callbackResponse.body, 'CMS site URL is not configured correctly.');
  } finally {
    if (previousOrigin === undefined) delete process.env.CMS_SITE_URL;
    else process.env.CMS_SITE_URL = previousOrigin;
  }
});
