export default function Footer() {
  return (
    <footer className="footer">
      <span>© {new Date().getFullYear()} Campus Safety Capstone</span>
      <a href="https://github.com/gsu-ds/campus-burglary-risk-prediction" target="_blank" rel="noreferrer">GitHub</a>
    </footer>
  );
}
