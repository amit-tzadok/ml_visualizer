import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PerceptronDemo from "./components/PerceptronDemo";
import CompareDemo from "./components/CompareDemo";
import "./App.css";

function App() {
  const [classifier, setClassifier] = useState("linear");
  const [compare, setCompare] = useState(false);
  const [runKey, setRunKey] = useState(0);
  const [theme, setTheme] = useState("light"); // "light" or "dark"

  const themes = {
    light: {
      background: "#f9f9f9",
      mainBackground: "#eef3ff",
      text: "#222",
      headerBg: "rgba(255,255,255,0.6)",
      controlBg: "rgba(255,255,255,0.9)",
      pointPositive: "#3b82f6", // blue
      pointNegative: "#ef4444", // red
      line: "#111"
    },
    dark: {
      background: "#1a1a1a",
      mainBackground: "#2c2c2c",
      text: "#f0f0f0",
      headerBg: "rgba(30,30,30,0.8)",
      controlBg: "rgba(40,40,40,0.9)",
      pointPositive: "#60a5fa", // lighter blue
      pointNegative: "#f87171", // lighter red
      line: "#f0f0f0"
    }
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
        transition: "background 0.3s ease, color 0.3s ease"
      }}
    >
      {/* Header */}
      <header
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: 60,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: currentTheme.headerBg,
          backdropFilter: "blur(4px)",
          color: currentTheme.text,
          zIndex: 1100,
          boxShadow: "0 1px 8px rgba(0,0,0,0.08)",
          transition: "background 0.3s ease, color 0.3s ease"
        }}
      >
        <h1 style={{ margin: 0, fontSize: 28 }}>ML Visualizer</h1>
      </header>

      {/* Visualization area */}
      <main
        style={{
          position: "absolute",
          top: 60,
          bottom: 0,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          background: currentTheme.mainBackground,
          transition: "background 0.3s ease"
        }}
      >
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center"
          }}
        >
          {compare ? (
            <CompareDemo
              key={runKey}
              leftType={classifier}
              rightType={classifier === "linear" ? "poly" : "linear"}
              theme={currentTheme} // pass colors for dark/light mode
            />
          ) : (
            <PerceptronDemo key={runKey} classifierType={classifier} theme={currentTheme} />
          )}
        </div>
      </main>

      {/* Floating controls */}
      <div
        style={{
          position: "fixed",
          bottom: 20,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 12,
          padding: "8px 12px",
          background: currentTheme.controlBg,
          borderRadius: 12,
          boxShadow: "0 6px 20px rgba(0,0,0,0.15)",
          zIndex: 1200,
          transition: "background 0.3s ease, color 0.3s ease"
        }}
      >
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 14 }}>Algorithm</span>
          <select
            value={classifier}
            onChange={(e) => setClassifier(e.target.value)}
            style={{ marginLeft: 8 }}
          >
            <option value="linear">Linear Perceptron</option>
            <option value="poly">Polynomial Perceptron</option>
          </select>
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={compare}
            onChange={(e) => setCompare(e.target.checked)}
          />
          <span style={{ fontSize: 14 }}>Compare side-by-side</span>
        </label>

        <button
          onClick={() => setRunKey((k) => k + 1)}
          style={{ padding: "8px 12px" }}
        >
          Run Again
        </button>

        {/* Light/Dark toggle */}
        <button
          onClick={() => setTheme(theme === "light" ? "dark" : "light")}
          style={{ padding: "8px 12px" }}
        >
          {theme === "light" ? "Dark Mode" : "Light Mode"}
        </button>
      </div>

      {/* Keyboard help box */}
      <div
        style={{
          position: "fixed",
          right: 16,
          bottom: 16,
          background: currentTheme.controlBg,
          padding: 12,
          borderRadius: 8,
          boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
          fontSize: 13,
          zIndex: 1200,
          maxWidth: 320,
          transition: "background 0.3s ease, color 0.3s ease"
        }}
      >
        <strong>Keyboard controls</strong>
        <div style={{ marginTop: 8, lineHeight: 1.4 }}>
          <div><code>Space</code> — pause / resume training</div>
          <div><code>b</code> — add a blue (positive) point</div>
          <div><code>r</code> — add a red (negative) point</div>
          <div><code>R</code> — reset dataset & model</div>
          <div><code>+</code> / <code>-</code> — adjust speed</div>
        </div>
      </div>
    </div>
  );
}

export default App;
