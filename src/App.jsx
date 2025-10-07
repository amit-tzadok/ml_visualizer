import React from 'react';
import PerceptronDemo from './components/PerceptronDemo';
import './App.css';

function App() {
  return (
    <div className="app">
      <header>
        <h1>ML Visualizer</h1>
        <p>Interactive visualization of machine learning classifiers</p>
      </header>
      <main>
        <PerceptronDemo />
      </main>
    </div>
  );
}

export default App;