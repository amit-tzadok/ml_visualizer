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

type Theme = "light" | "dark";

const App: React.FC = () => {
  const [classifier, setClassifier] = useState<string>("linear");
  const [compare, setCompare] = useState<boolean>(false);
  const [runKey, setRunKey] = useState<number>(0);
  const [theme, setTheme] = useState<Theme>("light");
  const [showWelcome, setShowWelcome] = useState<boolean>(() => {
    try {
      return localStorage.getItem("mlv_seenWelcome") === "1" ? false : true;
    } catch {
      return true;
    }
  });
  const [fx, setFx] = useState<boolean>(true);
  const [speedScale, setSpeedScale] = useState<number>(1);
  const [showSpeedPrompt, setShowSpeedPrompt] = useState<boolean>(false);
  const [pendingAction, setPendingAction] = useState<any>(null);
  const [promptSpeed, setPromptSpeed] = useState<number>(1);

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
            <div
              style={{
                fontSize: "2rem",
                color: currentTheme.text,
              }}
            >
              🧠
            </div>
            <h1
              style={{
                margin: 0,
                fontSize: 32,
                fontWeight: "700",
                color: currentTheme.text,
                letterSpacing: "-0.5px",
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
            paddingBottom: 120,
            paddingRight: 260,
            paddingLeft: 80,
          }}
        >
          {showWelcome ? (
            <WelcomeAny
              onChoose={(choice) => {
                // ask for speed before launching chosen demo
                setPendingAction({ type: "welcome", classifier: choice });
                setPromptSpeed(speedScale);
                setShowSpeedPrompt(true);
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
                  // ask for speed before switching demo
                  setPendingAction({ type: "classifier", classifier: choice });
                  setPromptSpeed(speedScale);
                  setShowSpeedPrompt(true);
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
              {showSpeedPrompt && (
                <span
                  style={{
                    marginLeft: 10,
                    fontSize: 12,
                    color: currentTheme.accent,
                    fontWeight: 600,
                  }}
                >
                  ← choose speed (bottom-left)
                </span>
              )}
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
                setPendingAction({ type: "compare", compare: next });
                setPromptSpeed(speedScale);
                setShowSpeedPrompt(true);
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
            {showSpeedPrompt && (
              <span
                style={{
                  marginLeft: 8,
                  fontSize: 12,
                  color: currentTheme.accent,
                  fontWeight: 600,
                }}
              >
                ← choose speed
              </span>
            )}
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

      {/* Speed prompt modal */}
      {showSpeedPrompt && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0,0,0,0.4)",
            zIndex: 1300,
          }}
        >
          <div
            style={{
              minWidth: 320,
              background: currentTheme.controlBg,
              padding: 20,
              borderRadius: 12,
              boxShadow: `0 16px 48px ${currentTheme.shadow}`,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8 }}>
              Set training speed
            </div>
            <div style={{ marginBottom: 12 }}>
              Select a global speed before continuing.
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <input
                id="global-speed-prompt"
                type="range"
                min="0.25"
                max="3"
                step="0.05"
                value={promptSpeed}
                onChange={(e) => setPromptSpeed(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <div style={{ width: 56, textAlign: "right" }}>
                {promptSpeed.toFixed(2)}x
              </div>
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "flex-end",
                gap: 8,
                marginTop: 14,
              }}
            >
              <button
                onClick={() => {
                  setShowSpeedPrompt(false);
                  setPendingAction(null);
                }}
                style={{ padding: "8px 12px", borderRadius: 8 }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  // apply speed and then action
                  setSpeedScale(promptSpeed);
                  if (pendingAction) {
                    if (pendingAction.type === "classifier") {
                      setClassifier(pendingAction.classifier);
                    } else if (pendingAction.type === "compare") {
                      setCompare(Boolean(pendingAction.compare));
                    } else if (pendingAction.type === "welcome") {
                      if (pendingAction.classifier)
                        setClassifier(pendingAction.classifier);
                      setShowWelcome(false);
                      try {
                        localStorage.setItem("mlv_seenWelcome", "1");
                      } catch {}
                    }
                  }
                  setPendingAction(null);
                  setShowSpeedPrompt(false);
                }}
                style={{ padding: "8px 12px", borderRadius: 8 }}
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Keyboard help box (changes when KNN selected) */}
      {!showWelcome && (
        <div
          style={{
            position: "fixed",
            right: 20,
            bottom: 20,
            background: currentTheme.controlBg,
            padding: 16,
            borderRadius: 12,
            boxShadow: `0 8px 32px ${currentTheme.shadow}`,
            backdropFilter: "blur(20px)",
            border: `1px solid ${currentTheme.shadow}`,
            fontSize: 14,
            zIndex: 1200,
            maxWidth: 280,
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
    </div>
  );
};

export default App;
