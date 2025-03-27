import { useState, useEffect } from "react";
import Confetti from "react-confetti";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import "./App.css";

function App() {
  const [username, setUsername] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [code, setCode] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [score, setScore] = useState(null);
  const [solution, setSolution] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [challenge, setChallenge] = useState("");
  const [challengeId, setChallengeId] = useState(null);
  const [solvedCount, setSolvedCount] = useState(0);
  const [totalScore, setTotalScore] = useState(0);
  const [gameTimeLeft, setGameTimeLeft] = useState(900);
  const [isGameOver, setIsGameOver] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [showLoginGlow, setShowLoginGlow] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [showScores, setShowScores] = useState(false);
  const [questionScores, setQuestionScores] = useState([]);

  const API_BASE_URL = "http://localhost:5001";

  const startGame = async () => {
    if (!username.trim()) {
      setError("Please enter a username!");
      return;
    }
    setShowLoginGlow(true);
    try {
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Registration failed");
      }
      
      const data = await response.json();
      setSessionId(data.session_id);
      
      setTimeout(() => {
        setIsPlaying(true);
        fetchChallenge(true);
        setShowLoginGlow(false);
      }, 1000);
    } catch (err) {
      setError(err.message);
    }
  };

  const fetchChallenge = async (isFirst = false, retryCount = 0) => {
    if (isGameOver) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/get_challenge`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          username,
          session_id: sessionId 
        }),
      });

      if (response.status === 401) {
        setError("Session expired. Please refresh.");
        setIsPlaying(false);
        return;
      }

      if (response.status === 404) {
        throw new Error("No challenges available at this time");
      }

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      if (data.completed && data.progress?.answered >= 10) {
        endGame(true);
        return;
      }

      setChallenge(data.challenge.code);
      setChallengeId(data.challenge.id);
      setFeedback(null);
      setScore(null);
      setSolution(null);
      setCode("");

      if (!isFirst) {
        setSolvedCount(prev => {
          const newCount = data.progress?.answered || prev + 1;
          return newCount;
        });
        setGameTimeLeft(Math.floor(data.time_remaining || gameTimeLeft));
      }
    } catch (err) {
      if (retryCount < 2) {
        setTimeout(() => fetchChallenge(isFirst, retryCount + 1), 1000);
      } else {
        setError(`Failed to load challenge: ${err.message}`);
        if (err.message.includes("No challenges available")) {
          endGame(false);
        }
      }
    } finally {
      setLoading(false);
    }
  };

  const evaluateCode = async () => {
    if (!code.trim()) {
      setError("Please enter code to evaluate!");
      return;
    }
    
    if (!code.includes("def ") || !code.includes("(") || !code.includes(")")) {
      setError("Code must contain a function definition");
      return;
    }
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          session_id: sessionId,
          code,
          challenge_code: challenge,
          challenge_id: challengeId,
        }),
      });
      
      if (response.status === 401) {
        setError("Session expired. Please refresh.");
        setIsPlaying(false);
        return;
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Evaluation failed");
      }
      
      const data = await response.json();
      setFeedback(data.feedback);
      setScore(data.score);
      setTotalScore(data.total_score);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchSolution = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/get_solution`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          username,
          session_id: sessionId,
          code: challenge,
        }),
      });
      
      if (response.status === 401) {
        setError("Session expired. Please refresh.");
        setIsPlaying(false);
        return;
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to fetch solution");
      }
      
      const data = await response.json();
      setSolution(data.solution || challenge);
    } catch (err) {
      setError(err.message);
      setSolution(challenge);
    } finally {
      setLoading(false);
    }
  };

  const fetchQuestionScores = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get_all_questions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          username,
          session_id: sessionId 
        }),
      });
      
      const data = await response.json();
      if (data.questions && players[username]) {
        const scores = data.questions.map(q => ({
          id: q.id,
          code: q.code,
          score: players[username].attempted_scores[q.id] || 0
        }));
        setQuestionScores(scores);
      }
    } catch (err) {
      setError("Failed to load question scores");
    }
  };

  const handleNextOrFinish = async () => {
    if (solvedCount >= 9) {
      endGame(true);
      return;
    }
    if (!feedback) {
      setError("Please evaluate code first!");
      return;
    }
    await fetchChallenge();
  };

  const endGame = (isComplete = false) => {
    setIsGameOver(true);
    if (isComplete && solvedCount >= 9) {
      setShowConfetti(true);
      setSolvedCount(10);
    }
  };

  useEffect(() => {
    if (!isPlaying || isGameOver) return;
    
    const interval = setInterval(() => {
      setGameTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          endGame(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    
    return () => clearInterval(interval);
  }, [isPlaying, isGameOver]);

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className={`app-container ${showLoginGlow ? 'login-glow' : ''}`}>
      {showConfetti && <Confetti recycle={false} numberOfPieces={500} />}
      
      {!isPlaying ? (
        <div className="login-wrapper">
          <div className="login-container glassmorphic">
            <h1 className="neon-title">REFACTOR ROYALE</h1>
            <div className="input-container">
              <input
                type="text"
                placeholder="ENTER USERNAME"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="neon-input"
                onKeyPress={(e) => e.key === 'Enter' && startGame()}
              />
              <span className="input-highlight"></span>
            </div>
            <button 
              onClick={startGame} 
              className="neon-button neon-button-cyan"
            >
              <span className="button-text">START CODING</span>
              <span className="button-glow"></span>
            </button>
            {error && <p className="error">{error}</p>}
          </div>
        </div>
      ) : (
        <div className="game-wrapper">
          <div className="game-container glassmorphic">
            <div className="game-header">
              <h1 className="neon-title">CODE OPTIMIZER</h1>
              <div className="stats-container">
                <div className="stat-box">
                  <span className="stat-label">PLAYER</span>
                  <span className="stat-value neon-cyan">{username}</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">SOLVED</span>
                  <span className="stat-value neon-purple">{solvedCount}/10</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">SCORE</span>
                  <span className="stat-value neon-green">{totalScore}</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">TIME</span>
                  <span className="stat-value neon-red">{formatTime(gameTimeLeft)}</span>
                </div>
              </div>
            </div>
            
            {!isGameOver ? (
              <>
                <div className="challenge-card glassmorphic">
                  <div className="card-header">
                    <h3>CHALLENGE</h3>
                    <div className="difficulty-meter">
                      {Array.from({ length: 5 }).map((_, i) => (
                        <span key={i} className={i < (solvedCount % 5) ? 'active' : ''}></span>
                      ))}
                    </div>
                  </div>
                  <pre className="neon-code">{challenge || "LOADING CHALLENGE..."}</pre>
                </div>
                
                <div className="editor-container">
                  <CodeMirror
                    value={code}
                    height="200px"
                    extensions={[python()]}
                    onChange={(value) => setCode(value)}
                    placeholder="// OPTIMIZE THIS CODE..."
                    theme="dark"
                  />
                </div>
                
                <div className="button-group">
                  <button
                    onClick={evaluateCode}
                    disabled={loading}
                    className="neon-button neon-button-blue"
                  >
                    OPTIMIZE
                  </button>
                  <button
                    onClick={handleNextOrFinish}
                    disabled={!feedback || loading}
                    className="neon-button neon-button-magenta"
                  >
                    {solvedCount >= 9 ? "FINISH" : "NEXT"}
                  </button>
                </div>
                
                {loading && <div className="loading-spinner"></div>}
                {error && <p className="error">{error}</p>}
                {feedback && (
                  <div className="feedback-card">
                    <h2>FEEDBACK</h2>
                    <p>{feedback}</p>
                    <button 
                      onClick={fetchSolution} 
                      className="neon-button neon-button-purple"
                    >
                      VIEW SOLUTION
                    </button>
                  </div>
                )}
                {solution && (
                  <div className="solution-card">
                    <h2>OPTIMAL SOLUTION</h2>
                    <pre>{solution}</pre>
                  </div>
                )}
              </>
            ) : (
              <div className="game-over">
                {solvedCount >= 10 ? (
                  <>
                    <h2 className="game-over-title neon-text-green">MISSION COMPLETE!</h2>
                    <div className="final-stats">
                      <div className="stat-item">
                        <span className="stat-label">FINAL SCORE:</span>
                        <span className="stat-value neon-cyan">{totalScore}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">CHALLENGES SOLVED:</span>
                        <span className="stat-value neon-purple">10/10</span>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <h2 className="game-over-title neon-text-red">TIME'S UP!</h2>
                    <div className="final-stats">
                      <div className="stat-item">
                        <span className="stat-label">FINAL SCORE:</span>
                        <span className="stat-value neon-cyan">{totalScore}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">CHALLENGES SOLVED:</span>
                        <span className="stat-value neon-purple">{solvedCount}/10</span>
                      </div>
                    </div>
                  </>
                )}
                
                <div className="button-group">
                  <button 
                    onClick={() => window.location.reload()} 
                    className="neon-button neon-button-blue"
                  >
                    NEW GAME
                  </button>
                  <button 
                    onClick={() => {
                      fetchQuestionScores();
                      setShowScores(!showScores);
                    }}
                    className="neon-button neon-button-magenta"
                  >
                    {showScores ? "HIDE SCORES" : "VIEW SCORES"}
                  </button>
                </div>
                
                {showScores && (
                  <div className="scores-container glassmorphic">
                    <h3>QUESTION SCORES</h3>
                    <div className="scores-grid">
                      {questionScores.map((q) => (
                        <div key={q.id} className="score-item">
                          <div className="score-question">
                            <span className="score-label">Q{q.id}:</span>
                            <pre className="score-code">{q.code}</pre>
                          </div>
                          <div className="score-value">
                            {q.score}/10
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;