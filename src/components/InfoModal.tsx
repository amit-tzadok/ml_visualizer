import React from "react";

type InfoModalProps = {
  isOpen: boolean;
  onClose: () => void;
  theme: any;
};

const InfoModal: React.FC<InfoModalProps> = ({ isOpen, onClose, theme }) => {
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.6)",
          backdropFilter: "blur(4px)",
          zIndex: 2000,
          cursor: "pointer",
        }}
      />
      
      {/* Modal */}
      <div
        style={{
          position: "fixed",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          background: theme.controlBg,
          color: theme.text,
          borderRadius: 16,
          padding: "32px",
          maxWidth: 680,
          maxHeight: "85vh",
          overflowY: "auto",
          zIndex: 2001,
          boxShadow: `0 20px 60px ${theme.shadow}`,
          border: `1px solid ${theme.shadow}`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: 28, fontWeight: 700, fontFamily: "'Audiowide', system-ui" }}>
              ML Visualizer
            </h2>
            <p style={{ margin: "8px 0 0 0", opacity: 0.7, fontSize: 14 }}>
              Interactive Machine Learning Algorithm Visualization
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              fontSize: 28,
              cursor: "pointer",
              color: theme.text,
              padding: 0,
              lineHeight: 1,
              opacity: 0.6,
            }}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Tech Stack */}
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Tech Stack</h3>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {["React", "TypeScript", "p5.js", "Vite"].map((tech) => (
              <span
                key={tech}
                style={{
                  background: theme.accent,
                  color: "#fff",
                  padding: "4px 12px",
                  borderRadius: 6,
                  fontSize: 13,
                  fontWeight: 500,
                }}
              >
                {tech}
              </span>
            ))}
          </div>
        </div>

        {/* Algorithms */}
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Algorithms</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <strong style={{ color: theme.accent }}>Linear Perceptron</strong>
              <p style={{ margin: "4px 0 0 0", fontSize: 14, opacity: 0.85 }}>
                Binary classification with linear decision boundary. Time complexity: O(n·e) where n = samples, e = epochs.
              </p>
            </div>
            <div>
              <strong style={{ color: theme.accent }}>Polynomial Perceptron</strong>
              <p style={{ margin: "4px 0 0 0", fontSize: 14, opacity: 0.85 }}>
                Non-linear classification using polynomial feature expansion (x, y, x², y², xy).
              </p>
            </div>
            <div>
              <strong style={{ color: theme.accent }}>Neural Network (MLP)</strong>
              <p style={{ margin: "4px 0 0 0", fontSize: 14, opacity: 0.85 }}>
                Multi-layer perceptron with customizable hidden layers, optimizers (SGD, Momentum, Adam).
              </p>
            </div>
            <div>
              <strong style={{ color: theme.accent }}>K-Nearest Neighbors</strong>
              <p style={{ margin: "4px 0 0 0", fontSize: 14, opacity: 0.85 }}>
                Non-parametric classification. Prediction complexity: O(n·d) per query.
              </p>
            </div>
          </div>
        </div>

        {/* Features */}
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Key Features</h3>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
            <li>Real-time visualization of decision boundaries during training</li>
            <li>Interactive point placement with instant model updates</li>
            <li>Performance optimizations: adaptive grid rendering, throttled repaints</li>
            <li>Compare mode: side-by-side algorithm comparison</li>
            <li>Maximize margin: automated hyperparameter sweep for optimal separation</li>
            <li>Adjustable training speed and customizable model parameters</li>
          </ul>
        </div>

        {/* About */}
        <div style={{ marginBottom: 0 }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>About</h3>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, opacity: 0.85 }}>
            Built as an educational tool to demonstrate core machine learning concepts through interactive visualization.
            All algorithms are implemented from scratch in TypeScript with no ML library dependencies.
          </p>
          <div style={{ marginTop: 16, display: "flex", gap: 12, alignItems: "center" }}>
            <a
              href="https://github.com/amit-tzadok/ml_visualizer"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: theme.accent,
                textDecoration: "none",
                fontSize: 14,
                fontWeight: 500,
              }}
            >
              View on GitHub →
            </a>
          </div>
        </div>
      </div>
    </>
  );
};

export default InfoModal;
