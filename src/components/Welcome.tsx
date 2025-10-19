import React from "react";

type WelcomeProps = {
  onChoose?: (choice: string) => void;
};

const Welcome: React.FC<WelcomeProps> = ({ onChoose }) => {
  const cards = [
    {
      key: "linear",
      title: "🎯 Linear Perceptron",
      desc: "Straight-line boundary; hinge-loss option",
    },
    {
      key: "poly",
      title: "🎯 Polynomial Perceptron",
      desc: "Non-linear boundary with xy, x², y²",
    },
    {
      key: "mlp",
      title: "🧠 Neural Network (MLP)",
      desc: "Learns decision regions epoch-by-epoch",
    },
    {
      key: "knn",
      title: "👥 K-Nearest Neighbors",
      desc: "Classify by nearest points; adjustable k",
    },
  ];
  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#fff",
        fontFamily: "system-ui, sans-serif",
        padding: "1rem",
        animation: "fadeIn 0.6s ease-out",
      }}
    >
      <div
        style={{
          maxWidth: "900px",
          textAlign: "center",
          background: "rgba(255,255,255,0.08)",
          backdropFilter: "blur(20px)",
          borderRadius: "24px",
          padding: "2rem",
          boxShadow: "0 20px 60px rgba(0,0,0,0.2)",
          border: "1px solid rgba(255,255,255,0.2)",
          animation: "slideUp 0.8s ease-out 0.2s both",
          position: "relative",
        }}
      >
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
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "1rem",
            marginBottom: "2rem",
          }}
        >
          {cards.map((card, idx) => {
            const tints: Record<string, any> = {
              linear: {
                tint1: "rgba(59,130,246,0.22)",
                tint2: "rgba(59,130,246,0.12)",
                glow: "rgba(59,130,246,0.45)",
                border: "rgba(59,130,246,0.45)",
              },
              poly: {
                tint1: "rgba(236,72,153,0.22)",
                tint2: "rgba(236,72,153,0.12)",
                glow: "rgba(236,72,153,0.45)",
                border: "rgba(236,72,153,0.45)",
              },
              mlp: {
                tint1: "rgba(139,92,246,0.22)",
                tint2: "rgba(139,92,246,0.12)",
                glow: "rgba(139,92,246,0.45)",
                border: "rgba(139,92,246,0.45)",
              },
              knn: {
                tint1: "rgba(6,182,212,0.22)",
                tint2: "rgba(6,182,212,0.12)",
                glow: "rgba(6,182,212,0.45)",
                border: "rgba(6,182,212,0.45)",
              },
            };
            const c = tints[card.key] || tints.linear;
            return (
              <button
                key={card.key}
                onClick={() => onChoose && onChoose(card.key)}
                style={{
                  padding: "1.4rem 1.2rem",
                  background: `linear-gradient(145deg, ${c.tint1}, ${c.tint2}), linear-gradient(145deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08))`,
                  borderRadius: "16px",
                  backdropFilter: "blur(20px)",
                  border: `1px solid ${c.border}`,
                  transition: "all 0.3s ease",
                  animation: `slideUp 0.8s ease-out ${1 + idx * 0.15}s both`,
                  textAlign: "left",
                  cursor: "pointer",
                  color: "#fff",
                  position: "relative",
                  overflow: "hidden",
                }}
                onMouseOver={(e) => {
                  (e.currentTarget as HTMLElement).style.transform = "translateY(-4px)";
                  (e.currentTarget as HTMLElement).style.boxShadow = `0 20px 50px ${c.glow}`;
                }}
                onMouseOut={(e) => {
                  (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                  (e.currentTarget as HTMLElement).style.boxShadow = "none";
                }}
              >
                <div style={{ fontSize: "1.1rem", fontWeight: 700 }}>{card.title}</div>
                <div style={{ fontSize: "0.9rem", opacity: 0.9 }}>{card.desc}</div>
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: "-100%",
                    width: "100%",
                    height: "100%",
                    background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent)",
                    transform: "skewX(-20deg)",
                    filter: "blur(2px)",
                    animation: "welcomeShine 2.8s ease-in-out infinite",
                  }}
                />
              </button>
            );
          })}
        </div>

        <div style={{ fontSize: "0.85rem", opacity: 0.8, marginTop: "8px", animation: "slideUp 0.8s ease-out 1.8s both" }}>
          Choose a card above to begin
        </div>
      </div>
    </div>
  );
};

export default Welcome;
