export default function Home() {
  return (
    <section>
      <h1>Predictive Analysis for Campus Safety</h1>
      <p>
        Modeling burglary risk across Atlanta universities (GSU, Georgia Tech, CAU, Spelman).
        This site hosts our methods, dataset notes, and live updates.
      </p>
      <ul className="cards">
        <li>
          <h3>Overview</h3>
          <p>Project goals, scope, and deliverables for the capstone.</p>
        </li>
        <li>
          <h3>Methods</h3>
          <p>CRISP-DM flow, features, models, evaluation metrics.</p>
        </li>
        <li>
          <h3>Dataset</h3>
          <p>Sources, time windows (2021–2023), pre-processing highlights.</p>
        </li>
      </ul>
    </section>
  );
}
