import React, { useState } from 'react';
import styles from './AgentPanel.module.css';

interface AgentPanelProps {
  classifier: string;
  compare: boolean;
  speedScale: number;
  theme: any;
  isOpen: boolean;
  onClose: () => void;
}

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const AgentPanel: React.FC<AgentPanelProps> = ({
  classifier,
  compare,
  speedScale,
  theme,
  isOpen,
  onClose,
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm your local ML Assistant. I can help you understand machine learning concepts, explain classifiers, and provide tips for using the visualizer. What would you like to know?",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');

  const getClassifierInfo = (type: string) => {
    const info: { [key: string]: { description: string; strengths: string[]; weaknesses: string[]; useCases: string[] } } = {
      linear: {
        description: "A linear perceptron that finds a straight line to separate data points into two classes.",
        strengths: ["Simple and fast", "Works well with linearly separable data", "Easy to understand"],
        weaknesses: ["Cannot model curved/complex boundaries", "Sensitive to outliers", "Underfits non-linear patterns"],
        useCases: ["Binary classification", "Well-separated data", "Educational purposes"],
      },
      poly: {
        description: "A polynomial perceptron that can learn curved decision boundaries using polynomial features.",
        strengths: ["Can handle non-linear data", "More flexible than linear", "Still relatively simple"],
        weaknesses: ["Feature explosion with high degrees", "Prone to overfitting without regularization", "Choice of degree is manual"],
        useCases: ["Non-linear classification", "Curved decision boundaries", "Moderate complexity data"],
      },
      mlp: {
        description: "A Multi-Layer Perceptron (neural network) with configurable hidden layers and neurons.",
        strengths: ["Very flexible", "Can learn complex patterns", "Modern deep learning approach"],
        weaknesses: ["Needs tuning (layers, lr, epochs)", "Slower to train", "Less interpretable"],
        useCases: ["Complex classification", "Pattern recognition", "Advanced ML demonstrations"],
      },
      knn: {
        description: "K-Nearest Neighbors algorithm that classifies points based on their closest neighbors.",
        strengths: ["No training phase", "Intuitive", "Non-parametric"],
        weaknesses: ["Prediction slows with many points", "Sensitive to feature scaling & noise", "Choice of K affects bias/variance"],
        useCases: ["Instance-based learning", "Small datasets", "Understanding distance-based classification"],
      },
    };
    return info[type] || { description: "Unknown classifier", strengths: [], weaknesses: [], useCases: [] };
  };

  const formatInfo = (type: string) => {
    const info = getClassifierInfo(type);
    const title = type.toUpperCase();
    const strengths = info.strengths.map((s) => `• ${s}`).join('\n');
    const weaknesses = info.weaknesses.map((w) => `• ${w}`).join('\n');
    const uses = info.useCases.map((u) => `• ${u}`).join('\n');
    return `**${title}**\n${info.description}\n\nStrengths:\n${strengths}\n\nLimitations:\n${weaknesses}\n\nBest for:\n${uses}`;
  };

  const generateResponse = (userInput: string): string => {
    const input = userInput.toLowerCase().trim();

    // Final equation request
    if (input.includes('equation') || input.includes('formula') || input.includes('boundary')) {
      if (compare) {
        return "In Compare mode there are two classifiers at once, so I can’t show a single equation. Turn off Compare to get the current classifier’s equation.";
      }
      try {
        const status = (window as any).mlvStatus as (undefined | { classifier?: string; equation?: string | null; updatedAt?: number });
        if (classifier === 'linear' || classifier === 'poly') {
          if (status && status.classifier === classifier && status.equation) {
            return `Current ${classifier.toUpperCase()} decision boundary:\n${status.equation}`;
          }
          return classifier === 'linear'
            ? "I’ll show the line equation once the model has trained a bit. Try pausing the demo (Space) to freeze the final weights."
            : "I’ll show the polynomial boundary once the model has trained a bit. Try pausing the demo (Space).";
        }
        if (classifier === 'mlp') {
          return "The MLP doesn’t have a simple closed-form equation. It’s a composition of learned nonlinear layers. I can summarize its architecture or last loss if you’d like.";
        }
        if (classifier === 'knn') {
          return "KNN has no equation—it classifies by the labels of the nearest neighbors in the dataset.";
        }
      } catch {
        // fall through to default
      }
    }

    // Classifier questions
    if (input.includes('what') && input.includes('classifier')) {
      return formatInfo(classifier);
    }

    // Pros/Cons / Advantages/Disadvantages
    if (
      input.includes('advantage') || input.includes('advantages') ||
      input.includes('pro') || input.includes('pros') ||
      input.includes('disadvantage') || input.includes('disadvantages') ||
      input.includes('con') || input.includes('cons') ||
      input.includes('tradeoff') || input.includes('trade-offs') || input.includes('limitations')
    ) {
      // If user mentions a specific classifier, use it; else use current
      const types = ['linear','poly','mlp','knn'];
      const mentioned = types.find(t => input.includes(t) || (t==='linear' && input.includes('perceptron')) || (t==='poly' && input.includes('polynomial')) || (t==='mlp' && input.includes('neural')) || (t==='knn' && input.includes('nearest')));
      return formatInfo(mentioned || classifier);
    }

    if (input.includes('explain') || input.includes('what is')) {
      if (input.includes('linear') || input.includes('perceptron')) {
        return "The **Linear Perceptron** is the simplest form of a neural network. It tries to find a straight line that best separates your data points. Think of it as drawing a line to divide red points from blue points. It's great for data that's already somewhat separated, but struggles with complex patterns.";
      }
      if (input.includes('polynomial') || input.includes('poly')) {
        return "The **Polynomial Perceptron** extends the linear version by adding curved features. Instead of just a straight line, it can create curved decision boundaries. This helps when your data forms curved patterns that a straight line can't separate well.";
      }
      if (input.includes('mlp') || input.includes('neural') || input.includes('network')) {
        return "The **Multi-Layer Perceptron (MLP)** is a modern neural network with multiple layers of neurons. It can learn very complex patterns and is the foundation of deep learning. You can adjust the number of hidden layers and neurons to control its complexity.";
      }
      if (input.includes('knn') || input.includes('nearest')) {
        return "The **K-Nearest Neighbors (KNN)** algorithm is different - it doesn't 'learn' in the traditional sense. When you want to classify a new point, it looks at the K closest points in your training data and votes on the most common class. It's simple but effective for understanding distance-based classification.";
      }
    }

    // Speed and controls
    if (input.includes('speed') || input.includes('slow') || input.includes('fast')) {
      return `Current speed is ${speedScale}x. You can adjust this using the speed slider in the bottom-left. Higher speeds make animations faster but might be harder to follow. Try 0.5x for detailed observation or 2x for quick demonstrations.`;
    }

    // Compare mode
    if (input.includes('compare')) {
      return `Compare mode lets you see two classifiers side-by-side. ${compare ? "It's currently enabled" : "It's currently disabled"}. When enabled, you'll see both algorithms working simultaneously on the same data. This is great for understanding the differences between classifiers!`;
    }

    // Data and interaction
    if (input.includes('data') || input.includes('points') || input.includes('add')) {
      return "To add data points: **Left-click** for Class A (red points) or **Right-click** for Class B (blue points). On touch devices, use the 'Touch tap adds' control. Try to create interesting patterns - some classifiers work better with certain data distributions!";
    }

    // Tips and suggestions
    if (input.includes('tip') || input.includes('help') || input.includes('how')) {
      const tips = [
        "Try the Linear Perceptron first - it's the simplest to understand",
        "For curved patterns, switch to Polynomial Perceptron",
        "Use Compare mode to see how different algorithms perform on the same data",
        "KNN works well with small, well-clustered datasets",
        "The MLP can learn very complex patterns but might be slower",
        "Experiment with different data patterns to see how each algorithm responds",
      ];
      return "💡 **Tip:** " + tips[Math.floor(Math.random() * tips.length)];
    }

    // Current state
    if (input.includes('status') || input.includes('current') || input.includes('state')) {
      return `**Current Status:**
• Classifier: ${classifier.toUpperCase()}
• Compare Mode: ${compare ? 'Enabled' : 'Disabled'}
• Speed: ${speedScale}x
• Theme: ${theme === 'dark' ? 'Dark' : 'Light'}

Ready to explore machine learning!`;
    }

    // Default responses
    if (input.includes('hello') || input.includes('hi') || input.includes('hey')) {
      return "Hello! I'm here to help you learn about machine learning algorithms. Ask me about classifiers, get tips, or ask for explanations!";
    }

    if (input.includes('thank')) {
      return "You're welcome! Happy learning! 🎓";
    }

    // Fallback
    return "I can help you with:\n• Explaining different classifiers (Linear, Polynomial, MLP, KNN)\n• Tips for using the visualizer\n• Understanding current settings\n• Guidance on adding data points\n\nTry asking 'explain linear perceptron' or 'give me a tip'!";

  };

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: messages.length + 1,
      text: inputValue,
      isUser: true,
      timestamp: new Date(),
    };

    const botResponse: Message = {
      id: messages.length + 2,
      text: generateResponse(inputValue),
      isUser: false,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage, botResponse]);
    setInputValue('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) {
    return null; // Don't render anything when closed
  }

  return (
    <div className={`${styles.agentSidebar} ${isOpen ? styles.open : ''}`}>
      <div className={styles.agentHeader}>
        <h3>ML Assistant</h3>
        <button
          onClick={onClose}
          className={styles.closeButton}
          title="Close Assistant"
        >
          ✕
        </button>
      </div>
      <div className={styles.agentContent}>
        <div className={styles.chatContainer}>
          <div className={styles.messages}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`${styles.message} ${message.isUser ? styles.userMessage : styles.botMessage}`}
              >
                <div className={styles.messageText}>
                  {message.text.split('\n').map((line, i) => (
                    <div key={i}>{line}</div>
                  ))}
                </div>
                <div className={styles.messageTime}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}
          </div>
          <div className={styles.inputContainer}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me about ML, classifiers, or get tips..."
              className={styles.chatInput}
            />
            <button
              onClick={handleSendMessage}
              className={styles.sendButton}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentPanel;