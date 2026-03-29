import React from "react";

interface Props {
  children: React.ReactNode;
  theme?: { text?: string; controlBg?: string; accent?: string; shadow?: string };
}
interface State { hasError: boolean; error: Error | null }

class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    const T = this.props.theme ?? {};
    return (
      <div style={{
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        height: "100%", gap: 12, padding: 32, textAlign: "center",
        color: T.text ?? "#2d3748", fontFamily: "system-ui, sans-serif",
      }}>
        <div style={{ fontSize: 40 }}>⚠️</div>
        <div style={{ fontSize: 18, fontWeight: 700 }}>Something went wrong</div>
        <div style={{ fontSize: 12, opacity: 0.6, maxWidth: 340, lineHeight: 1.6 }}>
          {this.state.error?.message ?? "An unexpected error occurred in this demo."}
        </div>
        <button
          onClick={() => this.setState({ hasError: false, error: null })}
          style={{
            marginTop: 8, padding: "9px 20px", borderRadius: 8, border: "none",
            background: T.accent ?? "#f43f5e", color: "#fff",
            fontSize: 13, fontWeight: 600, cursor: "pointer",
            boxShadow: `0 2px 8px ${T.shadow ?? "rgba(0,0,0,0.15)"}`,
          }}
        >
          Try again
        </button>
      </div>
    );
  }
}

export default ErrorBoundary;
