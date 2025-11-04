import { NavLink } from 'react-router-dom';
export default function Nav() {
  return (
    <header className="nav">
      <div className="brand">Campus Safety Capstone</div>
      <nav>
        <NavLink to="/" end>Home</NavLink>
        <NavLink to="/methods">Methods</NavLink>
        <NavLink to="/dataset">Dataset</NavLink>
        <NavLink to="/team">Team</NavLink>
      </nav>
    </header>
  );
}
