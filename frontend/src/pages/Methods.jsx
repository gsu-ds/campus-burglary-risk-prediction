export default function Methods() {
  return (
    <section>
      <h1>Methods</h1>
      <h2>CRISP-DM</h2>
      <p>Business understanding → data understanding → preparation → modeling → evaluation → deployment.</p>
      <h2>Features & Models</h2>
      <ul>
        <li>Spatiotemporal features: grids, POIs, calendar, lags.</li>
        <li>Baselines & ML: logistic regression, Random Forest, XGBoost.</li>
        <li>Metrics: ROC-AUC, PR-AUC, F1, calibration.</li>
      </ul>
      <h2>Reproducibility</h2>
      <ul>
        <li>Seeded splits, versioned data, config-driven runs.</li>
      </ul>
    </section>
  );
}
