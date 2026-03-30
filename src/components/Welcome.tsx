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
    {
      key: "logreg",
      title: "📈 Logistic Regression",
      desc: "Probabilistic linear classifier with sigmoid",
    },
    {
      key: "dtree",
      title: "🌿 Decision Tree",
      desc: "Axis-aligned splits; visualise the tree",
    },
    {
      key: "rforest",
      title: "🌲 Random Forest",
      desc: "Ensemble of trees; majority-vote boundary",
    },
    {
      key: "svm",
      title: "⚡ Support Vector Machine",
      desc: "Max-margin boundary; see support vectors",
    },
  ];
  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#fff",
        fontFamily: "system-ui, sans-serif",
        padding: "clamp(10px, 2.2vmin, 18px)",
        animation: "fadeIn 0.6s ease-out",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          width: "min(900px, 100%)",
          maxHeight: "calc(100vh - clamp(20px, 4.4vmin, 36px))",
          textAlign: "center",
          background: "rgba(255,255,255,0.08)",
          backdropFilter: "blur(20px)",
          borderRadius: "24px",
          padding: "clamp(14px, 2.6vmin, 22px)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.2)",
          border: "1px solid rgba(255,255,255,0.2)",
          animation: "slideUp 0.8s ease-out 0.2s both",
          position: "relative",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: "clamp(8px, 1.6vmin, 14px)",
          overflow: "hidden",
          boxSizing: "border-box",
        }}
      >
        <div
          style={{
            fontSize: "clamp(1.8rem, 3.6vmin, 2.6rem)",
            fontWeight: "bold",
            marginBottom: 0,
            color: "#fff",
            textShadow: "0 2px 10px rgba(0,0,0,0.3)",
            animation: "bounceIn 1s ease-out 0.4s both",
          }}
        >
          <img
            src="/logo.png"
            alt="ML Visualizer"
            style={{
              width: "clamp(64px, 9vmin, 96px)",
              height: "clamp(64px, 9vmin, 96px)",
              display: "block",
              margin: "0 auto",
            }}
          />
        </div>
        <h1
          style={{
            fontSize: "clamp(1.55rem, 3.1vmin, 2.05rem)",
            margin: 0,
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
            fontSize: "clamp(0.95rem, 2.1vmin, 1.1rem)",
            margin: 0,
            opacity: 0.95,
            lineHeight: 1.45,
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
            gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
            gap: "clamp(8px, 1.4vmin, 12px)",
            marginTop: "clamp(6px, 1.2vmin, 10px)",
          }}
        >
          {cards.map((card, idx) => {
            type Tint = { tint1: string; tint2: string; glow: string; border: string };
            const tints: Record<string, Tint> = {
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
              logreg: {
                tint1: "rgba(245,158,11,0.22)",
                tint2: "rgba(245,158,11,0.12)",
                glow: "rgba(245,158,11,0.45)",
                border: "rgba(245,158,11,0.45)",
              },
              dtree: {
                tint1: "rgba(34,197,94,0.22)",
                tint2: "rgba(34,197,94,0.12)",
                glow: "rgba(34,197,94,0.45)",
                border: "rgba(34,197,94,0.45)",
              },
              rforest: {
                tint1: "rgba(16,185,129,0.22)",
                tint2: "rgba(16,185,129,0.12)",
                glow: "rgba(16,185,129,0.45)",
                border: "rgba(16,185,129,0.45)",
              },
              svm: {
                tint1: "rgba(239,68,68,0.22)",
                tint2: "rgba(239,68,68,0.12)",
                glow: "rgba(239,68,68,0.45)",
                border: "rgba(239,68,68,0.45)",
              },
            };
            const c: Tint = tints[card.key] || tints.linear;
            return (
              <button
                key={card.key}
                onClick={() => onChoose && onChoose(card.key)}
                style={{
                  padding: "clamp(12px, 2.2vmin, 18px) clamp(12px, 2.0vmin, 16px)",
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
                <div style={{ fontSize: "clamp(0.95rem, 2.0vmin, 1.05rem)", fontWeight: 700 }}>
                  {card.title}
                </div>
                <div style={{ fontSize: "clamp(0.78rem, 1.7vmin, 0.9rem)", opacity: 0.9, lineHeight: 1.25 }}>
                  {card.desc}
                </div>
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

        <div
          style={{
            fontSize: "clamp(0.78rem, 1.6vmin, 0.85rem)",
            opacity: 0.8,
            marginTop: "clamp(2px, 0.8vmin, 8px)",
            animation: "slideUp 0.8s ease-out 1.8s both",
          }}
        >
          Choose a card above to begin
        </div>
      </div>
    </div>
  );
};

export default Welcome;
