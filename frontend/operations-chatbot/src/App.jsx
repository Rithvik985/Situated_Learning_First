import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import Chatbot from './components/chatbot'

function App() {

  return (
    <Router>
       <div >
          <Routes>
            <Route path="/" element={<Chatbot />} />
          </Routes>
        </div>
    </Router>
  );
}

export default App;
