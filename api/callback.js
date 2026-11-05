const { timingSafeEqual } = require('node:crypto');

const STATE_COOKIE = 'decap_oauth_state';
const VERIFIER_COOKIE = 'decap_oauth_verifier';
const COOKIE_PATH = '/api';

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

function clearCookie(name) {
  return `${name}=; Path=${COOKIE_PATH}; Max-Age=0; HttpOnly; Secure; SameSite=Lax`;
}

function equals(left, right) {
  if (!left || !right || left.length !== right.length) return false;
  return timingSafeEqual(Buffer.from(left), Buffer.from(right));
}

function callbackPage(origin, message) {
  const originLiteral = JSON.stringify(origin);
  const messageLiteral = JSON.stringify(message);

  return `<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>正在完成登录…</title></head><body><p>正在完成登录，请稍候…</p><script>
    (function () {
      var targetOrigin = ${originLiteral};
      var popupOpener = window.opener;
      var message = ${messageLiteral};
      if (!popupOpener) return;

      function deliver(event) {
        if (event.origin !== targetOrigin || event.source !== popupOpener || event.data !== 'authorizing:github') return;
        popupOpener.postMessage(message, targetOrigin);
        window.removeEventListener('message', deliver);
        window.setTimeout(function () { window.close(); }, 250);
      }

      window.addEventListener('message', deliver);
      popupOpener.postMessage('authorizing:github', targetOrigin);
    })();
  </script></body></html>`;
}

function sendPage(res, origin, message, statusCode = 200) {
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  return res.status(statusCode).send(callbackPage(origin, message));
}

module.exports = async function callback(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).send('Method Not Allowed');
  }

  let origin;
  try {
    origin = appOrigin();
  } catch {
    return res.status(500).send('CMS site URL is not configured correctly.');
  }
  const clearCookies = [clearCookie(STATE_COOKIE), clearCookie(VERIFIER_COOKIE)];
  res.setHeader('Set-Cookie', clearCookies);

  if (req.query.error) {
    return sendPage(res, origin, `authorization:github:error:${req.query.error_description || req.query.error}`, 400);
  }

  const state = req.query.state;
  const code = req.query.code;
  const receivedState = req.cookies && req.cookies[STATE_COOKIE];
  const verifier = req.cookies && req.cookies[VERIFIER_COOKIE];
  if (!code || !equals(state, receivedState) || !verifier) {
    return sendPage(res, origin, 'authorization:github:error:Invalid OAuth state.', 400);
  }

  const clientId = process.env.GITHUB_OAUTH_CLIENT_ID;
  const clientSecret = process.env.GITHUB_OAUTH_CLIENT_SECRET;
  if (!clientId || !clientSecret) {
    return sendPage(res, origin, 'authorization:github:error:GitHub OAuth is not configured.', 500);
  }

  try {
    const tokenResponse = await fetch('https://github.com/login/oauth/access_token', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        client_id: clientId,
        client_secret: clientSecret,
        code,
        redirect_uri: `${origin}/api/callback`,
        code_verifier: verifier,
      }),
    });
    const tokenData = await tokenResponse.json();
    if (!tokenResponse.ok || !tokenData.access_token) {
      const reason = tokenData.error_description || tokenData.error || 'Token exchange failed.';
      return sendPage(res, origin, `authorization:github:error:${reason}`, 502);
    }

    const payload = JSON.stringify({ token: tokenData.access_token, provider: 'github' });
    return sendPage(res, origin, `authorization:github:success:${payload}`);
  } catch {
    return sendPage(res, origin, 'authorization:github:error:Token exchange failed.', 502);
  }
};
