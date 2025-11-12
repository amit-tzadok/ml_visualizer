import React, { Suspense, useState } from "react";
// Lazy-load heavy components to reduce initial bundle and improve responsiveness
const PerceptronDemo = React.lazy(() => import("./components/PerceptronDemo"));
const CompareDemo = React.lazy(() => import("./components/CompareDemo"));
const KnnDemo = React.lazy(() => import("./components/KnnDemo"));
const MlpDemo = React.lazy(() => import("./components/MlpDemo"));
const Welcome = React.lazy(() => import("./components/Welcome"));
const AgentPanel = React.lazy(() => import("./components/AgentPanel"));
const InfoModal = React.lazy(() => import("./components/InfoModal"));
const KeyboardShortcutsModal = React.lazy(
  () => import("./components/KeyboardShortcutsModal")
);

// JS components are lazy-loaded; use a permissive generic component type to avoid `any`
const CompareAny = CompareDemo as React.LazyExoticComponent<
  React.ComponentType<Record<string, unknown>>
>;
const KnnAny = KnnDemo as React.LazyExoticComponent<
  React.ComponentType<Record<string, unknown>>
>;
const WelcomeAny = Welcome as React.LazyExoticComponent<
  React.ComponentType<Record<string, unknown>>
>;
import "./App.css";

type Theme = "light" | "dark";

const App: React.FC = () => {
  const [classifier, setClassifier] = useState<string>("linear");
  const [compare, setCompare] = useState<boolean>(false);
  const [runKey, _setRunKey] = useState<number>(0);
  const [theme, setTheme] = useState<Theme>("light");
  // gutterTop keeps the right gutter below the header so header buttons remain clickable
  const [_gutterTop, setGutterTop] = useState<number>(12);
  const [showWelcome, setShowWelcome] = useState<boolean>(() => {
    try {
      return localStorage.getItem("mlv_seenWelcome") === "1" ? false : true;
    } catch {
      return true;
    }
  });
  const [fx, setFx] = useState<boolean>(true);
  const [speedScale, setSpeedScale] = useState<number>(1);
  const [showAgent, setShowAgent] = useState<boolean>(false);
  // Training status badge
  const [trainingStatus, setTrainingStatus] = useState<
    "idle" | "training" | "complete"
  >("idle");
  const [accuracy, setAccuracy] = useState<number | null>(null);
  // Removed under-canvas equation bar
  // Audio unlock (for subtle chime)
  const audioCtxRef = React.useRef<AudioContext | null>(null);
  const [audioUnlocked, setAudioUnlocked] = useState<boolean>(false);
  // Compare mode tracking
  const compareRef = React.useRef<boolean>(compare);
  const compareFinishTrackerRef = React.useRef<Set<string>>(new Set());
  // New modals
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);
  const [showKeyboardModal, setShowKeyboardModal] = useState<boolean>(false);

  const themes = {
    light: {
      background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
      mainBackground: "linear-gradient(135deg, #ecececff 0%, #d76bdbff 100%)",
      text: "#2d3748",
      headerBg: "rgba(255,255,255,0.95)",
      controlBg: "rgba(255,255,255,0.98)",
      pointPositive: "#4299e1",
      pointNegative: "#e53e3e",
      line: "#2d3748",
      accent: "#667eea",
      shadow: "rgba(0,0,0,0.1)",
    },
    dark: {
      background: "linear-gradient(135deg, #1a202c 0%, #2d3748 100%)",
      mainBackground: "linear-gradient(135deg, #2d3748 0%, #1a202c 100%)",
      text: "#f7fafc",
      headerBg: "rgba(45,55,72,0.95)",
      controlBg: "rgba(45,55,72,0.98)",
      pointPositive: "#63b3ed",
      pointNegative: "#fc8181",
      line: "#f7fafc",
      accent: "#63b3ed",
      shadow: "rgba(0,0,0,0.3)",
    },
  } as const;

  const currentTheme = themes[theme];
  const isDark = theme === "dark";
  // Bottom panel theming
  const panelBg = isDark
    ? "linear-gradient(180deg, rgba(45,55,72,0.92), rgba(45,55,72,0.86))"
    : "linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.90))";
  const panelBorder = isDark
    ? "1px solid rgba(255,255,255,0.08)"
    : "1px solid rgba(0,0,0,0.08)";
  const panelShadowOuter = isDark
    ? "0 16px 48px rgba(0,0,0,0.45)"
    : "0 16px 48px rgba(20,20,20,0.12)";
  const panelShadowInner = isDark
    ? "inset 0 1px 0 rgba(255,255,255,0.06)"
    : "inset 0 1px 0 rgba(255,255,255,0.65)";
  const dividerColor = isDark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.08)";

  React.useEffect(() => {
    const compute = () => {
      // compute gutter top based on header bottom to avoid overlaying header
      try {
        const hdr = document.querySelector("header");
        if (hdr && hdr instanceof HTMLElement) {
          const r = hdr.getBoundingClientRect();
          setGutterTop(Math.max(0, Math.round(r.bottom + 8)));
        } else {
          setGutterTop(showWelcome ? 0 : 12);
        }
      } catch {
        // Ignore errors when computing header position
        setGutterTop(showWelcome ? 0 : 12);
      }
    };

    compute();
    window.addEventListener("resize", compute);
    const ro = new ResizeObserver(compute);
    const popup = document.querySelector(
      '[id*="copilot"], [class*="copilot"], [data-testid*="copilot"]'
    );
    if (popup) ro.observe(popup as Element);
    const hdr = document.querySelector("header");
    if (hdr) ro.observe(hdr as Element);
    return () => {
      window.removeEventListener("resize", compute);
      try {
        ro.disconnect();
      } catch {
        // Ignore errors when disconnecting ResizeObserver
      }
    };
  }, [showWelcome]);

  // keep a live ref of compare mode for the event handler (effect below has empty deps)
  React.useEffect(() => {
    compareRef.current = compare;
    // reset compare tracker when mode toggles
    compareFinishTrackerRef.current.clear();
  }, [compare]);

  // Unlock audio context on first user interaction; confetti is not blocked by this
  React.useEffect(() => {
    const unlock = () => {
      // Do not instantiate AudioContext here (causes cross-browser constructor typing) —
      // simply mark audio as unlocked so UI features depending on it can proceed.
      try {
        setAudioUnlocked(true);
      } catch {
        setAudioUnlocked(false);
      }
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
      window.removeEventListener("touchstart", unlock);
    };
    window.addEventListener("pointerdown", unlock, { once: true });
    window.addEventListener("keydown", unlock, { once: true });
    window.addEventListener("touchstart", unlock, { once: true });
    return () => {
      // remove the listeners without casting — the original function reference is sufficient
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
      window.removeEventListener("touchstart", unlock);
    };
  }, [showWelcome]);

  // Global keyboard shortcut for help modal disabled: users will use the Agent for questions
  // (Kept intentionally empty to avoid capturing '?' / Cmd-? which conflicts with typing or Agent usage.)
  // Intentionally no-op: previously had an empty effect which served no purpose and
  // generated lint noise. Removing it keeps hooks clean and avoids unnecessary
  // dependencies.

  // Listen for demo start/reset events
  React.useEffect(() => {
    const handleReset = () => {
      setTrainingStatus("training");
      setAccuracy(null);
    };
    window.addEventListener("mlv:demo-reset", handleReset);
    return () => window.removeEventListener("mlv:demo-reset", handleReset);
  }, []);

  // Listen for demo finished events - update status badge only
  React.useEffect(() => {
    const playChime = () => {
      try {
        const ac = audioCtxRef.current;
        if (!ac || !audioUnlocked) return;
        const now = ac.currentTime;
        const gain = ac.createGain();
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(0.2, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.5);

        const osc1 = ac.createOscillator();
        osc1.type = "sine";
        osc1.frequency.setValueAtTime(660, now);
        osc1.frequency.exponentialRampToValueAtTime(880, now + 0.25);
        const osc2 = ac.createOscillator();
        osc2.type = "triangle";
        osc2.frequency.setValueAtTime(990, now);
        osc2.frequency.exponentialRampToValueAtTime(1320, now + 0.25);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(ac.destination);
        osc1.start(now);
        osc2.start(now + 0.02);
        osc1.stop(now + 0.5);
        osc2.stop(now + 0.5);
      } catch {
        // ignore audio errors
      }
    };
    const handler = (e: Event) => {
      try {
        const anyE = e as CustomEvent<Record<string, unknown>>;
        const detail = (anyE.detail || {}) as Record<string, unknown>;
        const d = detail as Record<string, unknown>;
        const rawReason: string =
          typeof d.reason === "string" ? d.reason : "completed";
        const isCompleted =
          rawReason === "converged" ||
          rawReason === "max-epochs" ||
          rawReason === "train-complete" ||
          rawReason === "max-margin";
        if (!isCompleted) return;

        // Update status and accuracy
        setTrainingStatus("complete");
        if (typeof d.accuracy === "number") {
          setAccuracy(d.accuracy);
        }

        playChime();
      } catch {
        // swallow malformed events
      }
    };
    window.addEventListener(
      "mlv:demo-finished",
      handler as unknown as (e: Event) => void
    );
    return () => {
      window.removeEventListener(
        "mlv:demo-finished",
        handler as unknown as (e: Event) => void
      );
    };
  }, [audioUnlocked]);

  // fixed gutter on the right to prevent overlays from covering the demo
  const GUTTER_WIDTH = 400; // px reserved on the right for sidebar and popups (adjustable)

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        margin: 0,
        overflow: "hidden",
        fontFamily: "system-ui, sans-serif",
        position: "relative",
        background: currentTheme.background,
        color: currentTheme.text,
        transition: "background 0.3s ease, color 0.3s ease",
      }}
      data-fx={fx ? "on" : "off"}
    >
      {/* Header (hidden on Welcome) */}
      {!showWelcome && (
        <header
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            height: 70,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: currentTheme.headerBg,
            backdropFilter: "blur(20px)",
            color: currentTheme.text,
            zIndex: 1100,
            boxShadow: `0 4px 20px ${currentTheme.shadow}`,
            borderBottom: `1px solid ${currentTheme.shadow}`,
            transition: "all 0.3s ease",
          }}
        >
          {/* Home / Welcome button (top-left corner) */}
          <div
            style={{
              position: "absolute",
              left: 16,
              top: 14,
            }}
          >
            <button
              onClick={() => {
                setShowWelcome(true);
                // optionally reset compare and keep speedScale as-is
                setCompare(false);
              }}
              style={{
                padding: "8px 10px",
                fontSize: 18,
                background: currentTheme.controlBg,
                border: `1px solid ${currentTheme.shadow}`,
                borderRadius: 10,
                cursor: "pointer",
                color: currentTheme.text,
                transition: "all 0.2s ease",
                boxShadow: `0 2px 8px ${currentTheme.shadow}`,
              }}
              title="Go to Welcome"
              aria-label="Go to Welcome"
            >
              🏠
            </button>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ display: "flex", alignItems: "center" }}>
              <img
                src={`${import.meta.env.BASE_URL}logo.png`}
                alt="MLV"
                style={{ width: 48, height: 48, objectFit: "contain" }}
              />
            </div>
            <h1
              style={{
                margin: 0,
                fontSize: 32,
                fontWeight: "700",
                color: currentTheme.text,
                letterSpacing: "-0.5px",
                fontFamily: "'Audiowide', system-ui, sans-serif",
              }}
            >
              ML Visualizer
            </h1>
          </div>
          <div
            style={{
              position: "absolute",
              right: 20,
              display: "flex",
              gap: "12px",
              alignItems: "center",
            }}
          >
            <button
              onClick={() => setFx((v) => !v)}
              style={{
                padding: "8px 12px",
                fontSize: "14px",
                background: currentTheme.controlBg,
                border: `1px solid ${currentTheme.shadow}`,
                borderRadius: "8px",
                cursor: "pointer",
                color: currentTheme.text,
                transition: "all 0.2s ease",
                boxShadow: `0 2px 8px ${currentTheme.shadow}`,
              }}
              title="Toggle background visual effects"
            >
              {fx ? "✨ FX On" : "⚡ FX Off"}
            </button>
            <button
              onClick={() => setShowAgent((v) => !v)}
              style={{
                padding: "8px 12px",
                fontSize: "14px",
                background: showAgent
                  ? currentTheme.accent
                  : currentTheme.controlBg,
                border: `1px solid ${currentTheme.shadow}`,
                borderRadius: "8px",
                cursor: "pointer",
                color: showAgent ? "#fff" : currentTheme.text,
                transition: "all 0.2s ease",
                boxShadow: `0 2px 8px ${currentTheme.shadow}`,
              }}
              title="Toggle ML Assistant"
            >
              🤖 Agent
            </button>
            <button
              onClick={() => setShowInfoModal(true)}
              style={{
                padding: "8px 12px",
                fontSize: "14px",
                background: currentTheme.controlBg,
                border: `1px solid ${currentTheme.shadow}`,
                borderRadius: "8px",
                cursor: "pointer",
                color: currentTheme.text,
                transition: "all 0.2s ease",
                boxShadow: `0 2px 8px ${currentTheme.shadow}`,
              }}
              title="About & Info"
            >
              ℹ️ Info
            </button>
            <button
              onClick={() => setShowKeyboardModal(true)}
              style={{
                padding: "8px 12px",
                fontSize: "14px",
                background: currentTheme.controlBg,
                border: `1px solid ${currentTheme.shadow}`,
                borderRadius: "8px",
                cursor: "pointer",
                color: currentTheme.text,
                transition: "all 0.2s ease",
                boxShadow: `0 2px 8px ${currentTheme.shadow}`,
              }}
              title="Keyboard Shortcuts"
            >
              ⌨️
            </button>
            <button
              onClick={() => setTheme(theme === "light" ? "dark" : "light")}
              style={{
                padding: "8px 12px",
                fontSize: "14px",
                background: currentTheme.controlBg,
                border: `1px solid ${currentTheme.shadow}`,
                borderRadius: "8px",
                cursor: "pointer",
                color: currentTheme.text,
                transition: "all 0.2s ease",
                boxShadow: `0 2px 8px ${currentTheme.shadow}`,
              }}
            >
              {theme === "light" ? "🌙" : "☀️"}
            </button>
          </div>
        </header>
      )}

      {/* Visualization area */}
      <main
        style={{
          position: "absolute",
          top: showWelcome ? 0 : 70,
          bottom: 0,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          background: currentTheme.mainBackground,
          transition: "all 0.3s ease",
        }}
      >
        {/* Background FX layers */}
        {fx && (
          <>
            <div className="mlv-bg-orbs">
              <div className="mlv-orb orb1" />
              <div className="mlv-orb orb2" />
              <div className="mlv-orb orb3" />
            </div>
            <div className="mlv-bg-grain" />
            <div className="mlv-bg-vignette" />
          </>
        )}
        <div
          id="mlv-demo-area"
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            position: "relative",
            zIndex: 5,
            boxSizing: "border-box",
            // reserve space for floating controls; compact panel requires less padding
            // Was 160, reduce to 104 for a tighter layout while keeping clear separation
            paddingBottom: showWelcome ? 0 : 104,
            paddingRight: showWelcome ? 0 : 28,
            paddingLeft: showWelcome ? 0 : 28,
            transition: "padding 0.3s ease",
          }}
        >
          <Suspense
            fallback={
              <div style={{ color: currentTheme.text, opacity: 0.7 }}>
                Loading…
              </div>
            }
          >
            {showWelcome ? (
              <WelcomeAny
                onChoose={(choice) => {
                  // directly launch chosen demo at current speed
                  setClassifier(choice);
                  setShowWelcome(false);
                  try {
                    localStorage.setItem("mlv_seenWelcome", "1");
                  } catch {
                    // Ignore localStorage errors
                  }
                }}
              />
            ) : compare ? (
              <div
                style={{
                  width: "96%",
                  height: "96%",
                  maxWidth: 1400,
                  maxHeight: 1200,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <CompareAny
                  key={runKey}
                  leftType={classifier}
                  rightType={classifier === "linear" ? "poly" : "linear"}
                  theme={currentTheme as unknown as Record<string, unknown>}
                  speedScale={speedScale}
                />
              </div>
            ) : classifier === "knn" ? (
              <KnnAny
                key={runKey}
                theme={currentTheme as unknown as Record<string, unknown>}
                showInstructions={false}
                speedScale={speedScale}
              />
            ) : classifier === "mlp" ? (
              <div
                style={{
                  width: "96%",
                  height: "96%",
                  maxWidth: 1400,
                  maxHeight: 1200,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <MlpDemo
                  key={runKey}
                  // component accepts speedScale and optional props
                  showInstructions={false}
                  speedScale={speedScale}
                />
              </div>
            ) : (
              <div
                style={{
                  width: "96%",
                  height: "96%",
                  maxWidth: 1400,
                  maxHeight: 1200,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <PerceptronDemo
                  key={runKey}
                  classifierType={classifier}
                  showInstructions={false}
                  speedScale={speedScale}
                  dataset={undefined}
                  onDatasetChange={undefined}
                  resetToken={undefined}
                />
              </div>
            )}
          </Suspense>
        </div>
      </main>

      {/* Unified bottom control bar with adjustable vertical offset from canvas */}
      {!showWelcome && (
        <div
          style={{
            position: "fixed",
            // Compact panel with a slightly smaller stand-off from the bottom
            bottom: 30, // was 42
            left: "50%",
            transform: "translateX(-50%)",
            display: "flex",
            alignItems: "center",
            gap: 24, // was 54
            padding: "10px 18px", // was 20px 48px
            background: fx ? panelBg : currentTheme.controlBg,
            borderRadius: 14, // was 24
            boxShadow: fx
              ? `${panelShadowOuter}, ${panelShadowInner}`
              : `0 6px 16px ${currentTheme.shadow}`,
            backdropFilter: fx ? "saturate(120%) blur(22px)" : "blur(12px)",
            border: panelBorder,
            zIndex: 1250,
            transition: "all 0.3s ease",
          }}
        >
          {/* Algorithm selector */}
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 13,
              fontWeight: 600,
              color: currentTheme.text,
            }}
          >
            <span aria-hidden style={{ fontSize: 16, lineHeight: 1 }}>
              🎯
            </span>
            <span style={{ opacity: 0.9, whiteSpace: "nowrap" }}>
              Algorithm
            </span>
            <select
              value={classifier}
              onChange={(e) => setClassifier(e.target.value)}
              style={{
                padding: "8px 12px",
                borderRadius: 8,
                border: panelBorder,
                background: isDark
                  ? "linear-gradient(180deg, rgba(55,65,81,0.95), rgba(45,55,72,0.9))"
                  : "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,247,250,0.96))",
                color: currentTheme.text,
                fontSize: 13,
                cursor: "pointer",
                minWidth: 160,
                boxShadow: `${panelShadowInner}, 0 1px 8px ${currentTheme.shadow}`,
                outline: "none",
              }}
              onFocus={(e) =>
                (e.currentTarget.style.boxShadow = `0 0 0 3px ${
                  isDark ? "rgba(99,179,237,0.35)" : "rgba(102,126,234,0.35)"
                }, ${panelShadowInner}`)
              }
              onBlur={(e) =>
                (e.currentTarget.style.boxShadow = `${panelShadowInner}, 0 2px 10px ${currentTheme.shadow}`)
              }
              id="mlv-algo-select"
            >
              <option value="linear">Linear Perceptron</option>
              <option value="poly">Polynomial Perceptron</option>
              <option value="mlp">Neural Network (MLP)</option>
              <option value="knn">K-Nearest Neighbors</option>
            </select>
          </label>

          {/* divider */}
          <div
            aria-hidden
            style={{
              width: 1,
              height: 24,
              background: dividerColor,
              opacity: 1,
              borderRadius: 1,
            }}
          />

          {/* Compare mode toggle */}
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 13,
              fontWeight: 600,
              color: currentTheme.text,
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <span aria-hidden style={{ fontSize: 16, lineHeight: 1 }}>
              🔄
            </span>
            <span style={{ whiteSpace: "nowrap" }}>Compare</span>
            <input
              type="checkbox"
              checked={compare}
              onChange={(e) => setCompare(e.target.checked)}
              style={{
                width: 16,
                height: 16,
                cursor: "pointer",
                accentColor: currentTheme.accent,
              }}
              id="mlv-compare-checkbox"
            />
          </label>

          {/* divider */}
          <div
            aria-hidden
            style={{
              width: 1,
              height: 24,
              background: dividerColor,
              opacity: 1,
              borderRadius: 1,
            }}
          />

          {/* Speed slider */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: 13,
              fontWeight: 600,
              color: currentTheme.text,
              minWidth: 200,
            }}
          >
            <span aria-hidden style={{ fontSize: 16, lineHeight: 1 }}>
              ⏱️
            </span>
            <span style={{ opacity: 0.9, whiteSpace: "nowrap" }}>Speed</span>
            <input
              type="range"
              min="0.25"
              max="3"
              step="0.05"
              value={speedScale}
              onChange={(e) => setSpeedScale(Number(e.target.value))}
              style={{
                width: 140,
                WebkitAppearance: "none",
                height: 5,
                borderRadius: 999,
                background: isDark
                  ? "linear-gradient(90deg, rgba(102,126,234,0.5), rgba(118,75,162,0.4))"
                  : "linear-gradient(90deg, rgba(255,0,128,0.35), rgba(102,126,234,0.25))",
                outline: "none",
              }}
            />
            <span
              style={{
                width: 48,
                textAlign: "right",
                fontFamily: "monospace",
                fontSize: 12,
              }}
            >
              {speedScale.toFixed(2)}x
            </span>
          </div>

          {/* Training status (moved from separate floating badge) */}
          {classifier !== "knn" && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "8px 14px",
                background: isDark
                  ? "linear-gradient(180deg, rgba(45,55,72,0.92), rgba(45,55,72,0.86))"
                  : "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,247,250,0.96))",
                border: panelBorder,
                borderRadius: 12,
                boxShadow: `${panelShadowInner}, 0 6px 16px ${currentTheme.shadow}`,
                fontSize: 13,
                fontWeight: 600,
                color: currentTheme.text,
                minWidth: 140,
              }}
            >
              <span style={{ fontSize: 16 }}>
                {trainingStatus === "complete"
                  ? "🟢"
                  : trainingStatus === "training"
                  ? "🟡"
                  : "⚪"}
              </span>
              <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <div>
                  {trainingStatus === "complete"
                    ? "Complete"
                    : trainingStatus === "training"
                    ? "Training..."
                    : "Ready"}
                </div>
                {accuracy !== null && (
                  <div
                    style={{
                      fontSize: 11,
                      color: currentTheme.accent,
                      fontFamily: "monospace",
                    }}
                  >
                    {(accuracy * 100).toFixed(1)}% acc
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Speed prompt removed: actions apply immediately at current speedScale */}

      {/* Copilot Popup is now rendered inside the right gutter above */}

      {/* Agent Panel */}
      {!showWelcome && (
        <Suspense fallback={null}>
          <AgentPanel
            classifier={classifier}
            compare={compare}
            speedScale={speedScale}
            theme={currentTheme}
            isOpen={showAgent}
            onClose={() => setShowAgent(false)}
          />
        </Suspense>
      )}

      {/* Info Modal */}
      <Suspense fallback={null}>
        <InfoModal
          isOpen={showInfoModal}
          onClose={() => setShowInfoModal(false)}
          theme={currentTheme}
        />
      </Suspense>

      {/* Keyboard Shortcuts Modal */}
      <Suspense fallback={null}>
        <KeyboardShortcutsModal
          isOpen={showKeyboardModal}
          onClose={() => setShowKeyboardModal(false)}
          theme={currentTheme}
          classifier={classifier}
        />
      </Suspense>

      {/* Status badge removed; integrated into bottom control bar */}
    </div>
  );
};

export default App;
