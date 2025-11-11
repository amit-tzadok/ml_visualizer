import React from "react";

type ThemeShape = {
  controlBg?: string;
  text?: string;
  accent?: string;
  shadow?: string;
};

type KeyboardShortcutsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  theme: ThemeShape;
  classifier: string;
};

const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({
  isOpen,
  onClose,
  theme,
  classifier,
}) => {
  if (!isOpen) return null;

  const shortcuts = classifier === "knn"
    ? [
        { key: "A", description: "Add Class A point (red)" },
        { key: "B", description: "Add Class B point (blue)" },
        { key: "P", description: "Switch to prediction mode" },
        { key: "+/-", description: "Adjust k (neighbors)" },
      ]
    : [
        { key: "B", description: "Add blue point (Class A)" },
        { key: "R", description: "Add red point (Class B)" },
        { key: "Space", description: "Pause/Resume training" },
        { key: "+/-", description: "Increase/Decrease speed" },
      ];

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
          padding: "28px 32px",
          maxWidth: 480,
          zIndex: 2001,
          boxShadow: `0 20px 60px ${theme.shadow}`,
          border: `1px solid ${theme.shadow}`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600 }}>
            ⌨️ Keyboard Shortcuts
          </h2>
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

        {/* Shortcuts List */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {shortcuts.map(({ key, description }) => (
            <div
              key={key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 16,
              }}
            >
              <kbd
                style={{
                  background: theme.shadow,
                  padding: "6px 12px",
                  borderRadius: 6,
                  fontSize: 14,
                  fontWeight: 600,
                  fontFamily: "monospace",
                  minWidth: 60,
                  textAlign: "center",
                  border: `1px solid ${theme.text}20`,
                }}
              >
                {key}
              </kbd>
              <span style={{ fontSize: 14, opacity: 0.9 }}>{description}</span>
            </div>
          ))}
        </div>

        <div
          style={{
            marginTop: 24,
            paddingTop: 20,
            borderTop: `1px solid ${theme.shadow}`,
            fontSize: 13,
            opacity: 0.7,
            textAlign: "center",
          }}
        >
          Open this panel from the header (⌨️ button)
        </div>
      </div>
    </>
  );
};

export default KeyboardShortcutsModal;
