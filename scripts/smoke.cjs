const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:5173/';
  const outScreenshot = process.argv[3] || '/tmp/mlv_screenshot.png';
  const logs = [];
  let hadError = false;

  const browser = await puppeteer.launch({
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
    defaultViewport: { width: 1200, height: 900 }
  });

  try {
    const page = await browser.newPage();

    page.on('console', msg => {
      const text = `[console:${msg.type()}] ${msg.text()}`;
      logs.push(text);
      console.log(text);
    });
    page.on('pageerror', err => {
      const text = `[pageerror] ${err.message}`;
      logs.push(text);
      console.error(text);
      hadError = true;
    });
    page.on('requestfailed', req => {
      const text = `[requestfailed] ${req.url()} -> ${req.failure().errorText}`;
      logs.push(text);
      console.error(text);
      hadError = true;
    });

    console.log('Navigating to', url);
    const resp = await page.goto(url, { waitUntil: 'networkidle2', timeout: 15000 });
    if (resp) {
      console.log('HTTP status:', resp.status());
    } else {
      console.log('No response object from navigation');
      hadError = true;
    }

  // wait a bit for dynamic scripts to run (cross-version compatible)
  await page.waitForSelector('body', { timeout: 5000 }).catch(() => {});
  await page.evaluate(() => new Promise((res) => setTimeout(res, 500)));

    await page.screenshot({ path: outScreenshot, fullPage: true });
    console.log('Saved screenshot to', outScreenshot);

    // collect window errors if any
    const errors = await page.evaluate(() => (window.__mlv_test_errors__ || []));
    if (errors && errors.length) {
      console.error('Window errors:', errors);
      hadError = true;
    }

    // save logs to file
    fs.writeFileSync('/tmp/mlv_console.log', logs.join('\n'));
    console.log('Saved console log to /tmp/mlv_console.log');
  } catch (err) {
    console.error('Smoke test failed:', err && err.message ? err.message : err);
    hadError = true;
  } finally {
    await browser.close();
    process.exit(hadError ? 2 : 0);
  }
})();
