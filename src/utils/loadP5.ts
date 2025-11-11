// Lightweight runtime loader for p5.js from a CDN with a fallback to dynamic import.
// This prevents bundling p5 into the main app chunk and keeps first-load small.
export default async function loadP5(): Promise<unknown> {
  if (typeof window === 'undefined') {
    // server-side — fall back to dynamic import (won't run in SSR normally)
    return await import('p5');
  }

  // If p5 is already present on window (e.g., loaded by another demo), reuse it
  const w = window as unknown as { p5?: unknown };
  if (w.p5) return w.p5;

  // Try loading from CDN
  try {
    const version = '2.0.5'; // keep in sync with package.json dependency
    const src = `https://cdn.jsdelivr.net/npm/p5@${version}/lib/p5.min.js`;
    // If a script tag for this src already exists, wait for it to load
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      await new Promise<void>((resolve, reject) => {
        if ((existing as HTMLScriptElement).getAttribute('data-loaded') === '1') return resolve();
        existing.addEventListener('load', () => resolve());
        existing.addEventListener('error', () => reject(new Error('p5 CDN load failed')));
      });
      if (w.p5) return w.p5;
    }

    await new Promise<void>((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.async = true;
      s.defer = false;
      s.onload = () => {
        try { (s as HTMLScriptElement).setAttribute('data-loaded', '1'); } catch { /* ignore */ }
        resolve();
      };
      s.onerror = () => reject(new Error('p5 CDN load failed'));
      document.head.appendChild(s);
    });

    if (w.p5) return w.p5;
  } catch {
    // CDN load failed. In production we expect p5 to be provided via the CDN loader,
    // so avoid trying to bundle/import it — surface a clear error instead.
    const env = (typeof import.meta !== 'undefined'
      ? (import.meta as unknown as { env?: { PROD?: boolean } }).env
      : undefined) as { PROD?: boolean } | undefined;
    const isProd = !!(env && env.PROD);
    if (isProd) {
      throw new Error('Failed to load p5 from CDN in production');
    }

    // Development fallback: try a local dynamic import so dev servers continue to work.
    try {
      const mod = await import('p5');
      const maybeDefault = (mod as unknown as { default?: unknown }).default;
      return maybeDefault ?? mod;
    } catch {
      throw new Error('Failed to load p5 from CDN and local import failed');
    }
  }
}
