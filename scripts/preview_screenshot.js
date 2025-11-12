import puppeteer from 'puppeteer';

(async () => {
  try {
    const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 768 });
    const url = 'http://localhost:5111/';
    console.log('navigating to', url);
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
    await page.screenshot({ path: 'dist/preview.png', fullPage: true });
    console.log('screenshot saved to dist/preview.png');
    await browser.close();
    process.exit(0);
  } catch (err) {
    console.error('screenshot error:', err);
    process.exit(2);
  }
})();
