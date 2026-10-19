"use client";

import { useEffect, useState, type FormEvent } from "react";

type Analysis = {
  duration_seconds: number;
  analyzed_seconds: number;
  sample_rate: number;
  rms: number;
  zero_crossing_rate: number;
  spectral_centroid_hz: number;
  waveform_min: number[];
  waveform_max: number[];
  frequencies_hz: number[];
  spectrum_db: number[];
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState<"waveform" | "spectrum">("waveform");

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setError("");
    setAnalysis(null);
    if (file.size > 32 * 1024 * 1024) {
      setError("Choose an audio file up to 32 MiB.");
      return;
    }
    setBusy(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/analyze", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not read the audio file.");
      setAnalysis(data);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Audio analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  const spectrum = analysis?.spectrum_db ?? [];
  const floor = Math.min(...spectrum, -100);
  const ceiling = Math.max(...spectrum, floor + 1);
  const spectrumPoints = spectrum.map((v, i) => `${i * 800 / Math.max(spectrum.length - 1, 1)},${190 - (v - floor) / (ceiling - floor) * 170}`).join(" ");

  return (
    <main>
      <header>
        <a className="brand" href="/" aria-label="Music Atlas home"><span className="mark" aria-hidden="true">♫</span> Music Atlas</a>
        <span className="project-note">Music analysis lab</span>
      </header>
      <div className="workspace">
        <aside>
          <h2>From sound to data</h2>
          <p>We compare ways of representing music and explore what each one can reveal.</p>
          <nav aria-label="Sections"><a href="#analyze">Track analysis</a><a href="#representations">Audio representations</a></nav>
          <div className="research-note"><strong>Research dataset</strong><p>FMA-full</p><small>The catalogue, models and search will be available once the data is ready.</small></div>
        </aside>
        <div className="content">
          <section id="analyze">
            <div className="section-top"><div><h1>What does your sound look like?</h1><p>Upload a track, listen to it and take a closer look at its signal.</p></div><span className="badge">Signal analysis</span></div>
            <form onSubmit={submit} className="upload">
              <div><label htmlFor="audio">Choose an audio file</label><p>WAV, FLAC or MP3 · up to 32 MiB · first 30 seconds</p><input id="audio" type="file" accept="audio/*" disabled={busy} onChange={event => { setFile(event.target.files?.[0] ?? null); setAnalysis(null); setError(""); }} /></div>
              <button disabled={!file || busy} type="submit">{busy ? "Analyzing…" : "Analyze track"}</button>
            </form>
            {audioUrl && file && <audio aria-label="Listen to selected audio" controls src={audioUrl} />}
            <div aria-live="polite">{busy && <p className="status">Reading audio and calculating signal features.</p>}{error && <p className="error" role="alert">{error}</p>}</div>
            {analysis ? <>
              <div className="metrics">
                <div><span>Track duration</span><strong>{analysis.duration_seconds.toFixed(1)} s</strong></div>
                <div><span>RMS level</span><strong>{analysis.rms.toFixed(3)}</strong></div>
                <div><span>Spectral centroid</span><strong>{Math.round(analysis.spectral_centroid_hz)} Hz</strong></div>
                <div><span>Zero-crossing rate</span><strong>{analysis.zero_crossing_rate.toFixed(3)}</strong></div>
              </div>
              <div className="plot">
                <div className="plot-top"><h2>{view === "waveform" ? "Waveform" : "Average power spectrum"}</h2><div className="tabs" aria-label="Plot type"><button aria-pressed={view === "waveform"} onClick={() => setView("waveform")}>Waveform</button><button aria-pressed={view === "spectrum"} onClick={() => setView("spectrum")}>Spectrum</button></div></div>
                <svg viewBox="0 0 800 210" role="img" aria-label={view === "waveform" ? "Waveform amplitude envelope" : "Power spectrum by frequency"}>
                  {[40, 105, 170].map(y => <line key={y} x1="0" x2="800" y1={y} y2={y} stroke="#e4e8f2" />)}
                  {view === "waveform" ? analysis.waveform_min.map((v, i) => <line key={i} x1={i * 800 / 255} x2={i * 800 / 255} y1={105 - Math.max(-1, Math.min(1, v)) * 95} y2={105 - Math.max(-1, Math.min(1, analysis.waveform_max[i])) * 95} stroke="#6456d9" strokeWidth="2" />) : <polyline points={spectrumPoints} fill="none" stroke="#207d78" strokeWidth="2" />}
                </svg>
                <div className="axis"><span>0 {view === "waveform" ? "s" : "Hz"}</span><span>{view === "waveform" ? `${analysis.analyzed_seconds.toFixed(1)} s` : `${Math.round(analysis.frequencies_hz.at(-1) ?? 0)} Hz`}</span></div>
                <small>Analyzed segment: {analysis.analyzed_seconds.toFixed(1)} s. RMS describes signal level; the spectral centroid is the center of mass of its spectrum.</small>
              </div>
            </> : !busy && <div className="empty"><span aria-hidden="true">♪</span><h2>Start with one track</h2><p>Upload audio to see its waveform, spectrum and basic acoustic features.</p></div>}
          </section>
          <section id="representations" className="representations">
            <h2>Three ways to represent audio</h2><p>We plan to compare all three methods on the same set of tracks.</p>
            <div className="methods">{[
              ["Acoustic features", "MFCCs, chroma and rhythm", "A compact description based on measurable audio properties."],
              ["Log-mel spectrogram + CNN", "Frequency patterns over time", "A model that learns patterns from log-mel spectrograms."],
              ["Pretrained audio embeddings", "Vector representations", "A pretrained audio encoder maps tracks to vectors for classification and similarity search."],
            ].map(([title, subtitle, description]) => <article key={title}><small>Planned</small><h3>{title}</h3><p className="method-subtitle">{subtitle}</p><p>{description}</p></article>)}</div>
          </section>
          <footer>Music Atlas · Analysis of Unstructured Data · 2026</footer>
        </div>
      </div>
    </main>
  );
}
