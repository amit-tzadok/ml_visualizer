import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import loadP5 from "./utils/loadP5";

async function bootstrap() {
  // Ensure p5 is available before the app mounts. loadP5 will be a no-op if p5
  // is already present. In production this loads the CDN script; in dev it
  // falls back to a dynamic import so local development keeps working.
  try {
    await loadP5();
  } catch (err) {
    // If p5 fails to load in production, we still try to render the app so
    // the rest of the UI (docs, non-demo pages) remains usable. Log to console
    // for easier debugging of the deployed white-screen issue.
    console.warn("loadP5 failed:", err);
  }

  createRoot(document.getElementById("root") as HTMLElement).render(
    <StrictMode>
      <App />
    </StrictMode>
  );
}

void bootstrap();
