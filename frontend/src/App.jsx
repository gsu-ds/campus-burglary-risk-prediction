import { Routes, Route } from 'react-router-dom';
import Nav from './components/Nav.jsx';
import Footer from './components/Footer.jsx';
import Home from './pages/Home.jsx';
import Methods from './pages/Methods.jsx';
import Dataset from './pages/Dataset.jsx';
import Team from './pages/Team.jsx';
import NotFound from './pages/NotFound.jsx';
export default function App() {
  return (
    <div className="app">
      <Nav />
      <main className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/methods" element={<Methods />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/team" element={<Team />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}
