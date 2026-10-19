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
      setError("Wybierz plik o rozmiarze do 32 MiB.");
      return;
    }
    setBusy(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/analyze", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Nie udało się odczytać pliku.");
      setAnalysis(data);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Analiza nie powiodła się.");
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
        <a className="brand" href="/" aria-label="Music Atlas, strona główna"><span className="mark" aria-hidden="true">♫</span> Music Atlas</a>
        <span className="project-note">Laboratorium analizy muzyki</span>
      </header>
      <div className="workspace">
        <aside>
          <h2>Od dźwięku do danych</h2>
          <p>Porównujemy sposoby opisywania muzyki i sprawdzamy, co każdy z nich potrafi usłyszeć.</p>
          <nav aria-label="Sekcje"><a href="#analyze">Analiza nagrania</a><a href="#representations">Reprezentacje audio</a></nav>
          <div className="research-note"><strong>Zbiór badawczy</strong><p>FMA-full</p><small>Katalog, modele i wyszukiwanie pojawią się po przygotowaniu danych.</small></div>
        </aside>
        <div className="content">
          <section id="analyze">
            <div className="section-top"><div><h1>Jak wygląda Twój dźwięk?</h1><p>Wczytaj nagranie, odsłuchaj je i zajrzyj do środka sygnału.</p></div><span className="badge">Analiza sygnału</span></div>
            <form onSubmit={submit} className="upload">
              <div><label htmlFor="audio">Wybierz nagranie</label><p>WAV, FLAC lub MP3 · do 32 MiB · pierwsze 30 sekund</p><input id="audio" type="file" accept="audio/*" disabled={busy} onChange={event => { setFile(event.target.files?.[0] ?? null); setAnalysis(null); setError(""); }} /></div>
              <button disabled={!file || busy} type="submit">{busy ? "Analizuję…" : "Analizuj nagranie"}</button>
            </form>
            {audioUrl && file && <audio aria-label="Odsłuch wybranego nagrania" controls src={audioUrl} />}
            <div aria-live="polite">{busy && <p className="status">Odczytuję nagranie i obliczam cechy sygnału.</p>}{error && <p className="error" role="alert">{error}</p>}</div>
            {analysis ? <>
              <div className="metrics">
                <div><span>Długość nagrania</span><strong>{analysis.duration_seconds.toFixed(1)} s</strong></div>
                <div><span>Poziom RMS</span><strong>{analysis.rms.toFixed(3)}</strong></div>
                <div><span>Centroid widmowy</span><strong>{Math.round(analysis.spectral_centroid_hz)} Hz</strong></div>
                <div><span>Przejścia przez zero</span><strong>{analysis.zero_crossing_rate.toFixed(3)}</strong></div>
              </div>
              <div className="plot">
                <div className="plot-top"><h2>{view === "waveform" ? "Przebieg sygnału" : "Średnie widmo mocy"}</h2><div className="tabs" aria-label="Rodzaj wykresu"><button aria-pressed={view === "waveform"} onClick={() => setView("waveform")}>Sygnał</button><button aria-pressed={view === "spectrum"} onClick={() => setView("spectrum")}>Widmo</button></div></div>
                <svg viewBox="0 0 800 210" role="img" aria-label={view === "waveform" ? "Obwiednia amplitudy w czasie" : "Widmo mocy w funkcji częstotliwości"}>
                  {[40, 105, 170].map(y => <line key={y} x1="0" x2="800" y1={y} y2={y} stroke="#e4e8f2" />)}
                  {view === "waveform" ? analysis.waveform_min.map((v, i) => <line key={i} x1={i * 800 / 255} x2={i * 800 / 255} y1={105 - Math.max(-1, Math.min(1, v)) * 95} y2={105 - Math.max(-1, Math.min(1, analysis.waveform_max[i])) * 95} stroke="#6456d9" strokeWidth="2" />) : <polyline points={spectrumPoints} fill="none" stroke="#207d78" strokeWidth="2" />}
                </svg>
                <div className="axis"><span>0 {view === "waveform" ? "s" : "Hz"}</span><span>{view === "waveform" ? `${analysis.analyzed_seconds.toFixed(1)} s` : `${Math.round(analysis.frequencies_hz.at(-1) ?? 0)} Hz`}</span></div>
                <small>Analizowany fragment: {analysis.analyzed_seconds.toFixed(1)} s. RMS opisuje poziom sygnału, centroid jego środek ciężkości w widmie.</small>
              </div>
            </> : !busy && <div className="empty"><span aria-hidden="true">♪</span><h2>Zacznij od jednego nagrania</h2><p>Tutaj zobaczysz przebieg sygnału, widmo i podstawowe cechy akustyczne.</p></div>}
          </section>
          <section id="representations" className="representations">
            <h2>Trzy sposoby słuchania</h2><p>Plan porównania reprezentacji na wspólnym zbiorze utworów.</p>
            <div className="methods">{[
              ["Cechy akustyczne", "MFCC, chroma i rytm", "Muzyka opisana zestawem mierzalnych właściwości."],
              ["Spektrogram + CNN", "Częstotliwość w czasie", "Model uczący się wzorców na spektrogramach log-mel."],
              ["Gotowe embeddingi", "Wektory podobieństwa", "Reprezentacja z wytrenowanego wcześniej encodera audio."],
            ].map(([title, subtitle, description]) => <article key={title}><small>W przygotowaniu</small><h3>{title}</h3><p className="method-subtitle">{subtitle}</p><p>{description}</p></article>)}</div>
          </section>
          <footer>Music Atlas · Analysis of Unstructured Data · 2026</footer>
        </div>
      </div>
    </main>
  );
}
