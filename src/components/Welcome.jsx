import React from "react";

const Welcome = ({ onClose }) => {
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        height: "100vh",
        width: "100vw",
        background:
          "linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#fff",
        fontFamily: "system-ui, sans-serif",
        padding: "1rem",
        zIndex: 9999,
        animation: "fadeIn 0.6s ease-out",
      }}
    >
      <div
        style={{
          maxWidth: "900px",
          textAlign: "center",
          background: "rgba(255,255,255,0.1)",
          backdropFilter: "blur(20px)",
          borderRadius: "24px",
          padding: "2rem",
          boxShadow: "0 20px 60px rgba(0,0,0,0.2)",
          border: "1px solid rgba(255,255,255,0.2)",
          animation: "slideUp 0.8s ease-out 0.2s both",
          position: "relative",
        }}
      >
        {/* Close button in top-right corner */}
        <button
          onClick={() => {
            console.log("Close button clicked");
            onClose();
          }}
          style={{
            position: "absolute",
            top: "20px",
            right: "20px",
            width: "40px",
            height: "40px",
            borderRadius: "50%",
            border: "2px solid rgba(255,255,255,0.5)",
            background: "rgba(255,255,255,0.1)",
            color: "#fff",
            fontSize: "20px",
            fontWeight: "bold",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 0.3s ease",
            zIndex: 10,
          }}
          onMouseOver={(e) => {
            e.target.style.background = "rgba(255,255,255,0.2)";
            e.target.style.borderColor = "rgba(255,255,255,0.8)";
          }}
          onMouseOut={(e) => {
            e.target.style.background = "rgba(255,255,255,0.1)";
            e.target.style.borderColor = "rgba(255,255,255,0.5)";
          }}
          title="Skip welcome and start exploring"
        >
          ✕
        </button>
        <div
          style={{
            fontSize: "3rem",
            fontWeight: "bold",
            marginBottom: "1rem",
            color: "#fff",
            textShadow: "0 2px 10px rgba(0,0,0,0.3)",
            animation: "bounceIn 1s ease-out 0.4s both",
          }}
        >
          🧠
        </div>
        <h1
          style={{
            fontSize: "2.2rem",
            margin: "0 0 1rem 0",
            fontWeight: "300",
            color: "#fff",
            textShadow: "0 2px 10px rgba(0,0,0,0.3)",
            letterSpacing: "-1px",
            animation: "slideUp 0.8s ease-out 0.6s both",
          }}
        >
          ML Visualizer
        </h1>
        <p
          style={{
            fontSize: "1.2rem",
            marginBottom: "2rem",
            opacity: 0.95,
            lineHeight: 1.6,
            maxWidth: "600px",
            marginLeft: "auto",
            marginRight: "auto",
            animation: "slideUp 0.8s ease-out 0.8s both",
          }}
        >
          Explore how different machine learning algorithms learn from data.
          Watch models train in real-time, add points by clicking, and compare
          classifiers side-by-side.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "1rem",
            marginBottom: "2rem",
          }}
        >
          <div
            style={{
              padding: "1.5rem",
              background: "rgba(255,255,255,0.15)",
              borderRadius: "16px",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(255,255,255,0.2)",
              transition: "all 0.3s ease",
              animation: "slideUp 0.8s ease-out 1s both",
            }}
          >
            <h3
              style={{
                margin: "0 0 0.5rem 0",
                fontSize: "1.3rem",
                fontWeight: "600",
              }}
            >
              🎯 Perceptron
            </h3>
            <p
              style={{
                margin: 0,
                fontSize: "0.9rem",
                opacity: 0.9,
                lineHeight: 1.4,
              }}
            >
              Linear & polynomial boundaries with hinge loss optimization
            </p>
          </div>
          <div
            style={{
              padding: "1.5rem",
              background: "rgba(255,255,255,0.15)",
              borderRadius: "16px",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(255,255,255,0.2)",
              transition: "all 0.3s ease",
              animation: "slideUp 0.8s ease-out 1.2s both",
            }}
          >
            <h3
              style={{
                margin: "0 0 0.5rem 0",
                fontSize: "1.3rem",
                fontWeight: "600",
              }}
            >
              🧠 Neural Network
            </h3>
            <p
              style={{
                margin: 0,
                fontSize: "0.9rem",
                opacity: 0.9,
                lineHeight: 1.4,
              }}
            >
              Multi-layer perceptron with automatic training
            </p>
          </div>
          <div
            style={{
              padding: "1.5rem",
              background: "rgba(255,255,255,0.15)",
              borderRadius: "16px",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(255,255,255,0.2)",
              transition: "all 0.3s ease",
              animation: "slideUp 0.8s ease-out 1.4s both",
            }}
          >
            <h3
              style={{
                margin: "0 0 0.5rem 0",
                fontSize: "1.3rem",
                fontWeight: "600",
              }}
            >
              👥 K-Nearest Neighbors
            </h3>
            <p
              style={{
                margin: 0,
                fontSize: "0.9rem",
                opacity: 0.9,
                lineHeight: 1.4,
              }}
            >
              Instance-based learning with adjustable neighbors
            </p>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            gap: "1.5rem",
            justifyContent: "center",
            flexWrap: "wrap",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <button
            onClick={() => {
              console.log("Start Exploring clicked");
              onClose();
            }}
            style={{
              padding: "16px 32px",
              fontSize: "1.2rem",
              background: "linear-gradient(135deg, #fff 0%, #f8f9ff 100%)",
              color: "#667eea",
              border: "3px solid rgba(102, 126, 234, 0.4)",
              borderRadius: "20px",
              cursor: "pointer",
              fontWeight: "800",
              boxShadow: "0 12px 40px rgba(102, 126, 234, 0.5)",
              transition: "all 0.3s ease",
              animation:
                "slideUp 0.8s ease-out 1.6s both, buttonPulse 2s ease-in-out infinite 2s",
              textTransform: "uppercase",
              letterSpacing: "1px",
              position: "relative",
              overflow: "hidden",
              minWidth: "220px",
            }}
            onMouseOver={(e) => {
              e.target.style.transform = "translateY(-4px) scale(1.08)";
              e.target.style.boxShadow = "0 20px 50px rgba(102, 126, 234, 0.6)";
              e.target.style.borderColor = "rgba(102, 126, 234, 0.8)";
              e.target.querySelector("div").style.left = "100%";
            }}
            onMouseOut={(e) => {
              e.target.style.transform = "translateY(0) scale(1)";
              e.target.style.boxShadow = "0 12px 40px rgba(102, 126, 234, 0.5)";
              e.target.style.borderColor = "rgba(102, 126, 234, 0.4)";
              e.target.querySelector("div").style.left = "-100%";
            }}
          >
            <span
              style={{
                position: "relative",
                zIndex: 2,
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <span style={{ fontSize: "1.5em" }}>🚀</span>
              <span>START EXPLORING</span>
            </span>
            <div
              style={{
                position: "absolute",
                top: 0,
                left: "-100%",
                width: "100%",
                height: "100%",
                background:
                  "linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent)",
                transition: "left 0.5s ease",
              }}
            ></div>
          </button>

          <div
            style={{
              fontSize: "0.85rem",
              opacity: 0.8,
              marginTop: "8px",
              animation: "slideUp 0.8s ease-out 1.8s both",
            }}
          >
            Press the button above or click ✕ to continue
          </div>
        </div>
      </div>
    </div>
  );
};

export default Welcome;
