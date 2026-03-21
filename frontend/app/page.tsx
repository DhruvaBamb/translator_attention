"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";

export default function Home() {
  const [activeTab, setActiveTab] = useState("hi");
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState("Initializing...");

  const tabs = [
    { id: "hi", name: "English → Hindi", desc: "Machine Translation (Kaggle Dataset)" },
    { id: "es", name: "English → Spanish", desc: "Machine Translation (Opus Books)" },
    { id: "summary", name: "Summarization", desc: "Text Condensing (Real-world App)" },
  ];

  const handleAction = async () => {
    if (!inputText.trim()) return;
    setIsLoading(true);
    try {
      let endpoint = activeTab === "summary" ? "/summarize" : `/translate/${activeTab}`;
      let body = activeTab === "summary" ? { text: inputText } : { text: inputText };
      
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      setOutputText(data.translated_text || data.summary || "No output received.");
    } catch (err) {
      // Fallback for demo if backend is not running
      setOutputText(`[DEMO MODE] Could not connect to backend. Please ensure the FastAPI server is running at localhost:8000. \nInput: ${inputText}`);
    }
    setIsLoading(false);
  };

  return (
    <main style={{ minHeight: "100vh", padding: "2rem" }}>
      {/* Header */}
      <nav style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4rem" }}>
        <h2 className="gradient-text" style={{ fontSize: "1.5rem" }}>Translator AI</h2>
        <div style={{ display: "flex", gap: "2rem", color: "var(--text-dim)" }}>
          <span>Architecture</span>
          <span>Datasets</span>
          <span>Performance</span>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="animate-fade" style={{ display: "flex", gap: "4rem", alignItems: "center", marginBottom: "6rem" }}>
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: "4rem", marginBottom: "1.5rem", lineHeight: 1.1 }}>
            Mastering <span className="gradient-text">Sequence to Sequence</span> Deep Learning
          </h1>
          <p style={{ fontSize: "1.2rem", color: "var(--text-dim)", marginBottom: "2rem" }}>
            Explore the power of Encoder-Decoder architectures in Machine Translation, Summarization, and Speech Recognition. Our model bridges the gap across languages using contextualized representations.
          </p>
          <button className="cool-button" style={{ padding: "1rem 2.5rem", fontSize: "1.1rem" }}>
            View Implementation Plan
          </button>
        </div>
        <div style={{ flex: 1, position: "relative", height: "500px" }} className="glass">
          <Image 
            src="/hero.png" 
            alt="Neural Network Hero" 
            fill 
            style={{ objectFit: "cover", borderRadius: "1.5rem", opacity: 0.8 }} 
          />
        </div>
      </section>

      {/* Main Interface */}
      <section className="glass" style={{ padding: "3rem", marginBottom: "6rem" }}>
        <div style={{ textAlign: "center", marginBottom: "3rem" }}>
          <h2 style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>Model Playground</h2>
          <p style={{ color: "var(--text-dim)" }}>Interact with our various Seq2Seq implementations below.</p>
        </div>

        {/* Custom Tabs */}
        <div style={{ display: "flex", gap: "1rem", marginBottom: "2rem" }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setOutputText(""); }}
              style={{
                flex: 1, padding: "1.2rem", borderRadius: "1rem", border: "1px solid var(--border)",
                background: activeTab === tab.id ? "rgba(99, 102, 241, 0.1)" : "transparent",
                borderColor: activeTab === tab.id ? "var(--primary)" : "var(--border)",
                color: activeTab === tab.id ? "white" : "var(--text-dim)",
                cursor: "pointer", transition: "all 0.3s",
              }}
            >
              <div style={{ fontWeight: 700 }}>{tab.name}</div>
              <div style={{ fontSize: "0.8rem", opacity: 0.6 }}>{tab.desc}</div>
            </button>
          ))}
        </div>

        {/* Input/Output Group */}
        <div style={{ display: "flex", gap: "2rem" }}>
          <div style={{ flex: 1 }}>
            <h4 style={{ marginBottom: "0.5rem", color: "var(--text-dim)" }}>Source Input</h4>
            <textarea
              className="input-area"
              placeholder={activeTab === "summary" ? "Enter long text to summarize..." : "Enter text to translate..."}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
          </div>
          <div style={{ flex: 1 }}>
            <h4 style={{ marginBottom: "0.5rem", color: "var(--text-dim)" }}>Predicted Output</h4>
            <div className="input-area" style={{ background: "rgba(15, 23, 42, 0.8)", minHeight: "100px", color: "var(--foreground)" }}>
              {isLoading ? "Generating..." : outputText || "Waiting for input..."}
            </div>
          </div>
        </div>

        <div style={{ textAlign: "right", marginTop: "2rem" }}>
          <button className="cool-button" onClick={handleAction} disabled={isLoading}>
            {isLoading ? "Processing..." : "Run Inference"}
          </button>
        </div>
      </section>

      {/* Architecture Deep Dive */}
      <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3rem", marginBottom: "6rem" }}>
        <div className="glass" style={{ padding: "2rem" }}>
          <h3 style={{ marginBottom: "1rem" }} className="gradient-text">The Encoder</h3>
          <p style={{ color: "var(--text-dim)" }}>
            The encoder processes the input sequence and compresses it into a high-dimensional context vector. We utilize a multilayer LSTM architecture to capture long-range dependencies and contextual semantic relationships.
          </p>
          <ul style={{ marginTop: "1rem", listStyle: "none", color: "var(--text-dim)" }}>
            <li>• Embedding layer: Dimensionality Reduction</li>
            <li>• LSTM Stack: Hidden State extraction</li>
            <li>• Dropout (0.5): Regularization for stability</li>
          </ul>
        </div>
        <div className="glass" style={{ padding: "2rem" }}>
          <h3 style={{ marginBottom: "1rem" }} className="gradient-text">The Decoder</h3>
          <p style={{ color: "var(--text-dim)" }}>
            The decoder unrolls the context vector to generate the target sequence. It maintains a state of the previous predictions to ensure consistency across the generated text tokens.
          </p>
          <ul style={{ marginTop: "1rem", listStyle: "none", color: "var(--text-dim)" }}>
            <li>• Linear FC: Probability distribution over the vocabulary</li>
            <li>• Greedy/Beam Search decoding strategy</li>
            <li>• Cross-entropy optimized loss</li>
          </ul>
        </div>
      </section>

      {/* Performance Footer */}
      <footer style={{ textAlign: "center", padding: "4rem", borderTop: "1px solid var(--border)" }}>
        <p style={{ color: "var(--text-dim)" }}>
          Designed for Advanced Machine Translation Experiment | Translator AI Research v1.0
        </p>
      </footer>
    </main>
  );
}
