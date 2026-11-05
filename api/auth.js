const { createHash, randomBytes } = require('node:crypto');

const STATE_COOKIE = 'decap_oauth_state';
const VERIFIER_COOKIE = 'decap_oauth_verifier';
const COOKIE_PATH = '/api';
const MAX_AGE_SECONDS = 10 * 60;

function appOrigin() {
  const configuredOrigin = process.env.CMS_SITE_URL;
  if (!configuredOrigin) {
    throw new Error('CMS_SITE_URL must be configured.');
  }

  const origin = new URL(configuredOrigin);
  if (origin.pathname !== '/' || origin.search || origin.hash) {
    throw new Error('CMS_SITE_URL must contain only the site origin.');
  }
  return origin.origin;
}

function base64Url(buffer) {
  return buffer.toString('base64url');
}

function cookie(name, value, maxAge) {
  return `${name}=${value}; Path=${COOKIE_PATH}; Max-Age=${maxAge}; HttpOnly; Secure; SameSite=Lax`;
}

module.exports = function auth(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).send('Method Not Allowed');
  }

  const clientId = process.env.GITHUB_OAUTH_CLIENT_ID;
  if (!clientId) return res.status(500).send('GitHub OAuth is not configured.');

  try {
    const origin = appOrigin();
    const state = base64Url(randomBytes(32));
    const verifier = base64Url(randomBytes(48));
    const challenge = createHash('sha256').update(verifier).digest('base64url');
    const authorizeUrl = new URL('https://github.com/login/oauth/authorize');

    authorizeUrl.searchParams.set('client_id', clientId);
    authorizeUrl.searchParams.set('redirect_uri', `${origin}/api/callback`);
    authorizeUrl.searchParams.set('scope', 'public_repo');
    authorizeUrl.searchParams.set('state', state);
    authorizeUrl.searchParams.set('code_challenge', challenge);
    authorizeUrl.searchParams.set('code_challenge_method', 'S256');

    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Set-Cookie', [
      cookie(STATE_COOKIE, state, MAX_AGE_SECONDS),
      cookie(VERIFIER_COOKIE, verifier, MAX_AGE_SECONDS),
    ]);
    return res.redirect(302, authorizeUrl.toString());
  } catch (error) {
    return res.status(500).send(error.message);
  }
};
