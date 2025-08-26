import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import SituatedLearningApp from './components/situated.jsx'

function App() {

  return (
    <Router>
       <div >
          <Routes>
            <Route path="/" element={<SituatedLearningApp />} />
          </Routes>
        </div>
    </Router>
  );

}

export default App;
