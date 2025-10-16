import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PerceptronDemo from "./components/PerceptronDemo";
import CompareDemo from "./components/CompareDemo";
import KnnDemo from "./components/KnnDemo";
import MlpDemo from "./components/MlpDemo";
import Welcome from "./components/Welcome";
import "./App.css";

function App() {
  const [classifier, setClassifier] = useState("linear");
  const [compare, setCompare] = useState(false);
  const [runKey, setRunKey] = useState(0);
  const [theme, setTheme] = useState("light"); // "light" or "dark"
  const [showWelcome, setShowWelcome] = useState(true); // Always show welcome first

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
  };

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
    >
      {/* Header */}
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
          <button
            onClick={() => {
              try {
                localStorage.removeItem("mlv_welcome_shown");
              } catch (e) {}
              setShowWelcome(true);
            }}
            style={{
              padding: "6px 12px",
              fontSize: "12px",
              background: currentTheme.controlBg,
              border: `1px solid ${currentTheme.shadow}`,
              borderRadius: "6px",
              cursor: "pointer",
              color: currentTheme.text,
              transition: "all 0.2s ease",
              boxShadow: `0 2px 8px ${currentTheme.shadow}`,
            }}
          >
            🔄 Reset
          </button>
        </div>
      </header>

      {/* Visualization area */}
      <main
        style={{
          position: "absolute",
          top: 70,
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
        <div
          id="mlv-demo-area"
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          {compare ? (
            <CompareDemo
              key={runKey}
              leftType={classifier}
              rightType={classifier === "linear" ? "poly" : "linear"}
              theme={currentTheme}
            />
          ) : classifier === "knn" ? (
            <KnnDemo
              key={runKey}
              theme={currentTheme}
              showInstructions={false}
            />
          ) : classifier === "mlp" ? (
            <MlpDemo
              key={runKey}
              theme={currentTheme}
              showInstructions={false}
            />
          ) : (
            <PerceptronDemo
              key={runKey}
              classifierType={classifier}
              theme={currentTheme}
              showInstructions={false}
            />
          )}
        </div>
      </main>

      {/* Floating controls */}
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
          background: currentTheme.controlBg,
          borderRadius: 16,
          boxShadow: `0 8px 32px ${currentTheme.shadow}`,
          backdropFilter: "blur(20px)",
          border: `1px solid ${currentTheme.shadow}`,
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
              onChange={(e) => setClassifier(e.target.value)}
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

        <div
          style={{
            width: "1px",
            height: "24px",
            background: currentTheme.shadow,
            margin: "0 8px",
          }}
        ></div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
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
              onChange={(e) => setCompare(e.target.checked)}
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
      </div>

      {/* Keyboard help box (changes when KNN selected) */}
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
                  S
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
                  R
                </code>{" "}
                — Reset
              </div>
              <div>
                <code
                  style={{
                    background: currentTheme.shadow,
                    padding: "2px 4px",
                    borderRadius: 3,
                  }}
                >
                  K
                </code>{" "}
                — Change neighbors
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
                  R
                </code>{" "}
                — Reset
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
      {showWelcome && <Welcome onClose={() => setShowWelcome(false)} />}
    </div>
  );
}

export default App;
