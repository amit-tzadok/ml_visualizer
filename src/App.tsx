import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PerceptronDemo from "./components/PerceptronDemo";
import CompareDemo from "./components/CompareDemo";
import KnnDemo from "./components/KnnDemo";
import MlpDemo from "./components/MlpDemo";
import Welcome from "./components/Welcome";

// JS components are not typed yet; create any-typed aliases to avoid prop checks
const CompareAny: any = CompareDemo as any;
const KnnAny: any = KnnDemo as any;
// Use the typed MLP demo directly (TSX)
// Use the typed PerceptronDemo directly
const WelcomeAny: any = Welcome as any;
import "./App.css";
import { CopilotPopup } from "@copilotkit/react-ui";
import AgentPanel from "./components/AgentPanel";

type Theme = "light" | "dark";

const App: React.FC = () => {
  const [classifier, setClassifier] = useState<string>("linear");
  const [compare, setCompare] = useState<boolean>(false);
  const [runKey, setRunKey] = useState<number>(0);
  const [theme, setTheme] = useState<Theme>("light");
    // gutterTop keeps the right gutter below the header so header buttons remain clickable
    const [gutterTop, setGutterTop] = useState<number>(12);
  const [showWelcome, setShowWelcome] = useState<boolean>(() => {
    try {
      return localStorage.getItem("mlv_seenWelcome") === "1" ? false : true;
    } catch {
      return true;
    }
  });
  const [fx, setFx] = useState<boolean>(true);
  const [speedScale, setSpeedScale] = useState<number>(1);
  const [agentOpen, setAgentOpen] = useState<boolean>(false);

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

  // dynamic help box positioning to avoid overlapping header or side popups
  // now positioned at the bottom-right (left of the chatbot/copilot popup)
  const [helpPos, setHelpPos] = useState<{ bottom: number; right: number; zIndex: number }>({ bottom: 24, right: 20, zIndex: 1120 });

  React.useEffect(() => {
    const compute = () => {
      // position above the bottom (fixed offset)
      let bottom = 24;

      // default right offset
      let right = 20;
      let zIndex = 1120;

      // try to find a Copilot popup (common patterns) and avoid overlapping it
      const popup = document.querySelector('[id*="copilot"], [class*="copilot"], [data-testid*="copilot"]');
      if (popup && popup instanceof HTMLElement) {
        try {
          const pr = popup.getBoundingClientRect();
          // if the popup is on the right side, place help box to the left of it
          if (pr.left > window.innerWidth / 2) {
            // compute right so help box sits just left of popup with 12px gap
            right = Math.max(20, Math.round(window.innerWidth - pr.left + 12));
          }
          const popupZ = Number(window.getComputedStyle(popup).zIndex || 0) || 0;
          if (popupZ && !Number.isNaN(popupZ)) {
            zIndex = Math.max(1000, popupZ - 10);
          }
        } catch (e) {}
      }

      setHelpPos({ bottom, right, zIndex });
        // compute gutter top based on header bottom to avoid overlaying header
        try {
          const hdr = document.querySelector("header");
          if (hdr && hdr instanceof HTMLElement) {
            const r = hdr.getBoundingClientRect();
            setGutterTop(Math.max(0, Math.round(r.bottom + 8)));
          } else {
            setGutterTop(showWelcome ? 0 : 12);
          }
        } catch (e) {
          setGutterTop(showWelcome ? 0 : 12);
        }
    };

    compute();
    window.addEventListener("resize", compute);
    const ro = new ResizeObserver(compute);
    const popup = document.querySelector('[id*="copilot"], [class*="copilot"], [data-testid*="copilot"]');
    if (popup) ro.observe(popup as Element);
    const hdr = document.querySelector("header");
    if (hdr) ro.observe(hdr as Element);
    return () => {
      window.removeEventListener("resize", compute);
      try { ro.disconnect(); } catch (e) {}
    };
  }, [showWelcome]);

  // fixed gutter on the right to prevent overlays from covering the demo
  const GUTTER_WIDTH = 380; // px reserved on the right for popups (adjustable)

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
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <img src="/logo.png" alt="MLV" style={{ width: 48, height: 48, objectFit: 'contain' }} />
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
              onClick={() => setAgentOpen((v) => !v)}
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
              title="Open Assistant"
            >
              Assistant
            </button>
            <button
              onClick={() => {
                try { localStorage.removeItem('mlv_seenWelcome'); } catch {}
                setShowWelcome(true);
              }}
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
              title="Show Welcome Screen"
            >
              Welcome
            </button>
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
            // reserve space so floating controls don't overlap the demo canvas
            // when the welcome screen is shown we want a true centered layout,
            // so set padding to 0 to avoid visual offset caused by the demo padding.
            paddingBottom: showWelcome ? 0 : 120,
            // reserve a right gutter for side popups like Copilot/help
            paddingRight: showWelcome ? 0 : GUTTER_WIDTH,
            paddingLeft: showWelcome ? 0 : 80,
          }}
        >
          {showWelcome ? (
                <WelcomeAny
              onChoose={(choice) => {
                // directly launch chosen demo at current speed
                setClassifier(choice);
                setShowWelcome(false);
                try {
                  localStorage.setItem("mlv_seenWelcome", "1");
                } catch {}
              }}
            />
          ) : compare ? (
            <CompareAny
              key={runKey}
              leftType={classifier}
              rightType={classifier === "linear" ? "poly" : "linear"}
              theme={currentTheme as any}
              speedScale={speedScale}
            />
          ) : classifier === "knn" ? (
            <KnnAny
              key={runKey}
              theme={currentTheme as any}
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
        </div>
      </main>

      {/* interactive right gutter that contains Copilot/help popups so they can
          open fully without being cut off or overlapped by controls */}
      {!showWelcome && (
        <div
          id="mlv-right-gutter"
          style={{
            position: "fixed",
            top: gutterTop,
            right: 0,
            width: GUTTER_WIDTH,
            bottom: 0,
            pointerEvents: "auto",
            zIndex: 1350, // above floating controls (1200) and help box
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "flex-start",
            padding: 12,
            boxSizing: "border-box",
            overflow: "auto",
            WebkitOverflowScrolling: "touch",
          }}
        >
          <CopilotPopup
            instructions={
          `Assistant system context:
          ML Visualizer is a client-side React app that demonstrates simple ML classifiers (Linear Perceptron, Polynomial Perceptron, MLP, KNN).
          Key files:
          - demos: src/components/* (PerceptronDemo, MlpDemo, KnnDemo, CompareDemo)
          - models: src/utils/* (mlp.ts, perceptron.js, knn.js, polynomialClassifier.js)

          Common actions:
          - Algorithm dropdown switches demo (label: 'Algorithm').
          - Add points by clicking canvas: Left = class A (red), Right = class B (blue). On touch devices select 'Touch tap adds' in the Model Controls panel.
          - Floating panels: Speed (bottom-left), Algorithm+Compare (bottom-center), Model Controls (bottom-right), Help/Controls (top-right).
          - Useful keys: 'B' adds blue, 'R' adds red, Space toggles play/pause, +/- adjusts speed.

          Answer style:
          - Prefer short, precise, step-by-step answers for how to run or reproduce things locally.
          - When asking about code, mention the likely file and functions to check (e.g., 'See src/utils/mlp.ts -> MLPClassifier.fit').

          Example Q/A:
          Q: How do I retrain the MLP with a higher learning rate?
          A: Open Model Controls (bottom-right) -> set LR to 0.5 -> click Train (top-left of canvas) or call Train button inside the demo.

          If the user asks for code changes, suggest a minimal patch and test steps.
          `
            }
/>
        </div>
      )}

      {/* Floating controls (algorithm + compare) */}
      {!showWelcome && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            left: "50%",
            transform: "translateX(-50%)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: 16,
            padding: "16px 24px",
            background: fx ? currentTheme.controlBg : "rgba(0,0,0,0.02)",
            borderRadius: 16,
            boxShadow: fx
              ? `0 8px 32px ${currentTheme.shadow}`
              : `0 2px 6px ${currentTheme.shadow}`,
            backdropFilter: fx ? "blur(20px)" : "none",
            border: fx ? `1px solid ${currentTheme.shadow}` : "none",
            zIndex: 1200,
            transition: "all 0.3s ease",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 14,
                fontWeight: "500",
                color: currentTheme.text,
              }}
            >
              <span>🎯</span>
              <span>Algorithm</span>
              <select
                value={classifier}
                  onChange={(e) => {
                  const choice = e.target.value;
                  // switch demo immediately at current speed
                  setClassifier(choice);
                }}
                style={{
                  padding: "8px 12px",
                  borderRadius: "8px",
                  border: `1px solid ${currentTheme.shadow}`,
                  background: currentTheme.controlBg,
                  color: currentTheme.text,
                  fontSize: "14px",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                  boxShadow: `0 2px 8px ${currentTheme.shadow}`,
                }}
                id="mlv-algo-select"
              >
                <option value="linear">Linear Perceptron</option>
                <option value="poly">Polynomial Perceptron</option>
                <option value="mlp">Neural Network (MLP)</option>
                <option value="knn">K-Nearest Neighbors</option>
              </select>
              
            </label>
          </div>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 14,
              fontWeight: "500",
              color: currentTheme.text,
              cursor: "pointer",
            }}
          >
              <input
              type="checkbox"
              checked={compare}
              onChange={(e) => {
                const next = e.target.checked;
                // toggle compare immediately at current speed
                setCompare(next);
              }}
              style={{
                width: "18px",
                height: "18px",
                cursor: "pointer",
                accentColor: currentTheme.accent,
              }}
              id="mlv-compare-checkbox"
            />
            <span>🔄 Compare Mode</span>
            
          </label>
        </div>
      )}

      {/* Separate global speed control (its own panel) */}
      {!showWelcome && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            left: 24,
            zIndex: 1200,
            padding: "12px 16px",
            borderRadius: 12,
            background: currentTheme.controlBg,
            boxShadow: `0 8px 32px ${currentTheme.shadow}`,
            display: "flex",
            alignItems: "center",
            gap: 12,
          }}
        >
          <span style={{ fontSize: 14 }}>⏱️ Speed</span>
          <input
            type="range"
            min="0.25"
            max="3"
            step="0.05"
            value={speedScale}
            onChange={(e) => setSpeedScale(Number(e.target.value))}
            style={{ width: 140 }}
          />
          <div style={{ width: 48, textAlign: "right" }}>
            {speedScale.toFixed(2)}x
          </div>
        </div>
      )}

      {/* Speed prompt removed: actions apply immediately at current speedScale */}

      {/* Keyboard help box (changes when KNN selected) - positioned bottom-right, left of chatbot icon */}
      {!showWelcome && (
        <div
          style={{
            position: "fixed",
            right: helpPos.right,
            bottom: helpPos.bottom,
            background: currentTheme.controlBg,
            padding: 16,
            borderRadius: 12,
            boxShadow: `0 8px 32px ${currentTheme.shadow}`,
            backdropFilter: "blur(20px)",
            border: `1px solid ${currentTheme.shadow}`,
            fontSize: 14,
            zIndex: helpPos.zIndex,
            maxWidth: 320,
            transition: "all 0.3s ease",
          }}
          id="mlv-help-box"
        >
          <div
            style={{
              fontWeight: "600",
              marginBottom: 8,
              color: currentTheme.accent,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            {/* <span>⌨️</span> */}
            <span>Controls</span>
          </div>
          <div style={{ lineHeight: 1.6, color: currentTheme.text }}>
            {classifier === "knn" ? (
              <>
                <div>
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    A
                  </code>{" "}
                  — Add Class A point (red)
                </div>
                <div>
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    B
                  </code>{" "}
                  — Add Class B point (blue)
                </div>
                <div>
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    P
                  </code>{" "}
                  — Switch to prediction mode, then click to classify a new
                  point
                </div>
                <div style={{ marginTop: 6, opacity: 0.9 }}>
                  Use the k control in the KNN panel to adjust neighbors.
                </div>
              </>
            ) : (
              <>
                <div>
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    B
                  </code>{" "}
                  — Add blue point
                </div>
                <div>
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    R
                  </code>{" "}
                  — Add red point
                </div>
                <div>
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    Space
                  </code>{" "}
                  — Pause/Resume
                </div>

                <div>
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    +
                  </code>
                  /
                  <code
                    style={{
                      background: currentTheme.shadow,
                      padding: "2px 4px",
                      borderRadius: 3,
                    }}
                  >
                    -
                  </code>{" "}
                  — Speed
                </div>
              </>
            )}
          </div>
        </div>
      )}
  {/* Copilot Popup is now rendered inside the right gutter above */}

  {/* AgentPanel (local retrieval demo) */}
  <AgentPanel open={agentOpen} onClose={() => setAgentOpen(false)} />
    </div>
  );
};

export default App;
