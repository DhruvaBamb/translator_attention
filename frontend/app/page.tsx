"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";

export default function Home() {
  const [activeTab, setActiveTab] = useState("hi");
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState("Initializing...");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const tabs = [
    { id: "hi", name: "English → Hindi", desc: "Machine Translation (Kaggle Dataset)" },
    { id: "es", name: "English → Spanish", desc: "Machine Translation (Opus Books)" },
    { id: "fr", name: "English → French", desc: "Machine Translation (Opus Books)" },
    { id: "summary", name: "Summarization", desc: "Text Condensing (Real-world App)" },
    { id: "caption", name: "Image Captioning", desc: "Vision-to-Language (BLIP Model)" },
  ];

  const handleAction = async () => {
    if (activeTab !== "caption" && !inputText.trim()) return;
    if (activeTab === "caption" && !selectedImage) {
      alert("Please select an image first.");
      return;
    }

    setIsLoading(true);
    try {
      let response;
      if (activeTab === "caption" && selectedImage) {
        const formData = new FormData();
        formData.append("file", selectedImage);
        response = await fetch("http://localhost:8000/caption", {
          method: "POST",
          body: formData,
        });
      } else {
        const endpoint = activeTab === "summary" ? "/summarize" : `/translate/${activeTab}`;
        response = await fetch(`http://localhost:8000${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: inputText }),
        });
      }

      const data = await response.json();
      setOutputText(data.translated_text || data.summary || data.caption || "No output received.");
    } catch (err) {
      setOutputText(`[DEMO MODE] Could not connect to backend. Please ensure the FastAPI server is running at localhost:8000.`);
    }
    setIsLoading(false);
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
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
            {activeTab === "caption" ? (
              <div 
                className="input-area" 
                style={{ 
                  display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", 
                  gap: "1rem", border: "2px dashed var(--border)", cursor: "pointer", position: "relative"
                }}
                onClick={() => document.getElementById('imageInput')?.click()}
              >
                {imagePreview ? (
                  <img src={imagePreview} alt="Preview" style={{ maxWidth: "100%", maxHeight: "200px", borderRadius: "0.5rem" }} />
                ) : (
                  <div style={{ textAlign: "center" }}>
                    <p>Click to upload image</p>
                    <p style={{ fontSize: "0.8rem", opacity: 0.5 }}>PNG, JPG or WebP</p>
                  </div>
                )}
                <input 
                  id="imageInput" 
                  type="file" 
                  hidden 
                  accept="image/*" 
                  onChange={handleImageChange} 
                />
              </div>
            ) : (
              <textarea
                className="input-area"
                placeholder={activeTab === "summary" ? "Enter long text to summarize..." : "Enter text to translate..."}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
              />
            )}
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
          <h3 style={{ marginBottom: "1rem" }} className="gradient-text">
            {activeTab === "caption" ? "Vision Encoder (CNN/ViT)" : "Text Encoder (LSTM/Transformer)"}
          </h3>
          <p style={{ color: "var(--text-dim)" }}>
            {activeTab === "caption" 
              ? "The vision encoder (using ResNet or ViT) extracts high-level spatial features from the input image. These features are then projected into a shared embedding space for the decoder."
              : "The encoder processes the input sequence and compresses it into a high-dimensional context vector. We utilize specialized layers to capture long-range dependencies."}
          </p>
          <ul style={{ marginTop: "1rem", listStyle: "none", color: "var(--text-dim)" }}>
            {activeTab === "caption" ? (
              <>
                <li>• Convolutional Layers: Feature extraction</li>
                <li>• Global Average Pooling: Spatial reduction</li>
                <li>• Linear Projection: Hidden state alignment</li>
              </>
            ) : (
              <>
                <li>• Embedding layer: Dimensionality Reduction</li>
                <li>• Hidden State extraction: capturing context</li>
                <li>• Regularization: Ensuring generalization</li>
              </>
            )}
          </ul>
        </div>
        <div className="glass" style={{ padding: "2rem" }}>
          <h3 style={{ marginBottom: "1rem" }} className="gradient-text">The Decoder</h3>
          <p style={{ color: "var(--text-dim)" }}>
            {activeTab === "summary" 
              ? "The decoder generates a condensed version of the input, focusing on the most salient information using attention mechanisms (like in T5)."
              : "The decoder unrolls the context vector to generate the target sequence (text/caption). It maintains state consistency across generated tokens."}
          </p>
          <ul style={{ marginTop: "1rem", listStyle: "none", color: "var(--text-dim)" }}>
            <li>• Softmax layer: Vocab probability distribution</li>
            <li>• Auto-regressive decoding strategy</li>
            <li>• Contextualized output generation</li>
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
