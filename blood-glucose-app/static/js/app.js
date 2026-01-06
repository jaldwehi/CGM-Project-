const startBtn = document.getElementById("startBtn");
const predictBtn = document.getElementById("predictBtn");
const backHomeBtn = document.getElementById("backHomeBtn");
const againBtn = document.getElementById("againBtn");

const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");

const after1hEl = document.getElementById("after1h");
const riskLabelEl = document.getElementById("riskLabel");
const llmAdviceEl = document.getElementById("llmAdvice");

let chartInstance = null;

function scrollToId(id) {
  document.getElementById(id).scrollIntoView({ behavior: "smooth" });
}

function setLoading(isLoading, msg = "Loading forecast...") {
  loading.textContent = msg;
  loading.classList.toggle("hidden", !isLoading);
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function formatShortTime(iso) {
  // "2026-01-04T12:35:00" -> "12:35"
  return iso.slice(11, 16);
}

function drawChart(history, forecast) {
  const labels = [...history.times, ...forecast.times].map(formatShortTime);
  const values = [...history.values, ...forecast.values];

  const nowIndex = history.values.length - 1;

  const ctx = document.getElementById("chart").getContext("2d");

  if (chartInstance) chartInstance.destroy();

  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Real + Forecast (next 1 hour)",
          data: values,
          tension: 0.25,
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: { enabled: true },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 12 },
        },
      },
    },
    plugins: [
      {
        id: "nowLine",
        afterDraw(chart) {
          const { ctx, chartArea, scales } = chart;
          const x = scales.x.getPixelForValue(nowIndex);
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(x, chartArea.top);
          ctx.lineTo(x, chartArea.bottom);
          ctx.strokeStyle = "rgba(0,0,0,0.35)";
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.fillStyle = "rgba(0,0,0,0.6)";
          ctx.fillText("NOW", x + 6, chartArea.top + 14);
          ctx.restore();
        },
      },
    ],
  });
}

async function fetchForecast() {
  clearError();
  setLoading(true);

  try {
    const res = await fetch("/api/v1/forecast_hour", { cache: "no-store" });
    if (!res.ok) throw new Error(`Forecast failed: HTTP ${res.status}`);
    const data = await res.json();

    // Fill summary
    after1hEl.textContent = data.summary.after_1h.toFixed(1) + " mg/dL";

    // Draw chart
    drawChart(data.history, data.forecast);

    // Determine trend from history
    const h = data.history.values;
    let trend = "stable";
    if (h.length >= 2) {
      const delta = h[h.length - 1] - h[0];
      if (delta > 10) trend = "rising";
      else if (delta < -10) trend = "falling";
    }

    // Call LLM recommendation (separate request)
    setLoading(true, "Generating recommendation...");
    llmAdviceEl.textContent = "--";
    riskLabelEl.textContent = "--";

    const recRes = await fetch("/api/v1/recommendation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        predicted_glucose: data.summary.after_1h,
        trend,
      }),
    });

    if (!recRes.ok) throw new Error(`Recommendation failed: HTTP ${recRes.status}`);
    const rec = await recRes.json();

    riskLabelEl.textContent = rec.risk;
    llmAdviceEl.textContent = rec.advice;

    scrollToId("results");
  } catch (e) {
    showError(String(e));
  } finally {
    setLoading(false);
  }
}

startBtn.addEventListener("click", () => scrollToId("predict"));
backHomeBtn.addEventListener("click", () => scrollToId("home"));
againBtn.addEventListener("click", () => scrollToId("predict"));

predictBtn.addEventListener("click", fetchForecast);
