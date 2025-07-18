import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Maximize2, Minimize2, Send, User, Bot } from 'lucide-react';
import '../styles/chatbot.css';

function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [isWide, setIsWide] = useState(false);
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hello! How can I help you today?' }
  ]);

const BACKEND_SERVICE_URL = import.meta.env.VITE_BACKEND_SERVICE_URL;

  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messageContainerRef = useRef(null);
  const chatbotRef = useRef(null);

  const toggleChatbot = () => setIsOpen(!isOpen);
  const toggleSize = () => setIsWide(!isWide);

  const scrollToBottom = () => {
    const container = messageContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  };

  // Auto-scroll whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Additional auto-scroll during streaming
  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        scrollToBottom();
      }, 100); // Scroll every 100ms during streaming
      
      return () => clearInterval(interval);
    }
  }, [isStreaming]);

  // Click outside to close chatbot
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isOpen && chatbotRef.current && !chatbotRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const newMessage = { from: 'user', text: input };
    setMessages(prev => [...prev, newMessage]);
    const userQuery = input;
    setInput('');

    // Add bot message with typing indicator for streaming
    setMessages(prev => [...prev, { from: 'bot', text: '', isThinking: true }]);
    setIsStreaming(true);

    try {
      const response = await fetch(`${BACKEND_SERVICE_URL}/operations_chatbot/get_solution`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userquery: userQuery })
      });

      if (!response.ok || !response.body) {
        throw new Error('Failed to stream response');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;

        const chunk = decoder.decode(value || new Uint8Array(), { stream: !done });

        setMessages(prev => {
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          updated[lastIndex] = {
            ...updated[lastIndex],
            text: updated[lastIndex].text + chunk,
            isThinking: false // Remove thinking indicator when we start getting text
          };
          return updated;
        });

        // Force immediate scroll after each chunk
        setTimeout(scrollToBottom, 0);
      }

    } catch (error) {
      console.error('Error fetching response:', error);
      setMessages(prev => {
        const updated = [...prev];
        updated.pop();
        updated.push({ from: 'bot', text: 'Oops! Something went wrong. Please try again later.' });
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="chatbot-container" ref={chatbotRef}>
      <button className="chatbot-icon" onClick={toggleChatbot}>
        {isOpen ? <X size={24} /> : <MessageCircle size={24} />}
      </button>

      {isOpen && (
        <div className={`chatbot-window ${isWide ? 'wide' : ''}`}>
          <div className="chatbot-header">
            <div className="header-controls">
              <button className="header-button" onClick={toggleSize}>
                {isWide ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
              </button>
            </div>
            <h3 className="header-title">
              <Bot size={20} />
              AI Assistant
            </h3>
          </div>

          <div className="chatbot-body">
            <div className="message-container" ref={messageContainerRef}>
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.from === 'user' ? 'user-message' : 'bot-message'}`}>
                  <div className={`message-avatar ${msg.from === 'user' ? 'user-avatar' : 'bot-avatar'}`}>
                    {msg.from === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className={`message-bubble ${msg.from === 'user' ? 'user-bubble' : 'bot-bubble'}`}>
                    {msg.isThinking ? (
                      <div className="typing-indicator">
                        <span className="dot"></span>
                        <span className="dot"></span>
                        <span className="dot"></span>
                      </div>
                    ) : (
                      msg.text
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="chatbot-input">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message..."
              className="input-field"
              rows="1"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="send-button"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Chatbot;