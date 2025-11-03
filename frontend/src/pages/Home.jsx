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
        <li>
          <h3>Downloads</h3>
          <p>
            <a href="https://raw.githubusercontent.com/gsu-ds/campus-burglary-risk-prediction/main/CLEANED_Atlanta_Burglary_Larceny_All.csv" target="_blank" rel="noreferrer">Atlanta Burglary+Larceny (All).csv</a><br/>
            <a href="https://raw.githubusercontent.com/gsu-ds/campus-burglary-risk-prediction/main/CLEANED_GSU_Burglary_Larceny_1mi.csv" target="_blank" rel="noreferrer">GSU Burglary+Larceny 1mi.csv</a>
          </p>
        </li>
      </ul>
    </section>
  );
}
