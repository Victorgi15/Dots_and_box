import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const BOARD_SIZES = [3, 4, 5, 6, 7, 8];
const PLAYER_OPTIONS = [
  { value: "human", label: "Player" },
  { value: "chatgpt", label: "ChatGPT bot" },
  { value: "neural", label: "Neural bot" },
  { value: "random", label: "Random bot" },
  { value: "mcts", label: "MCTS bot" },
];
const BOT_DELAY_MS = 300;

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data.error || `Request failed (${response.status}).`;
    const error = new Error(message);
    error.data = data;
    throw error;
  }
  return data;
}

function winnerMessage(state) {
  if (!state || !state.done) {
    return "";
  }
  const s1 = state.scores["1"];
  const s2 = state.scores["2"];
  if (s1 > s2) {
    return "Game over. Player 1 wins.";
  }
  if (s2 > s1) {
    return "Game over. Player 2 wins.";
  }
  return "Game over. Draw.";
}

export default function App() {
  const [size, setSize] = useState(5);
  const [player1Type, setPlayer1Type] = useState("human");
  const [player2Type, setPlayer2Type] = useState("human");
  const [gameId, setGameId] = useState(null);
  const [state, setState] = useState(null);
  const [status, setStatus] = useState(
    "Choose players, then start a new game."
  );
  const [busy, setBusy] = useState(false);
  const [botHold, setBotHold] = useState(false);
  const [lastMove, setLastMove] = useState(null);

  const getPlayerType = (playerNumber) =>
    playerNumber === 1 ? player1Type : player2Type;

  const applyMoveResponse = (data, moveOverride = null) => {
    setState(data.state);
    setLastMove(moveOverride || data.move || null);
    setBotHold(false);
    if (data.state.done) {
      setStatus(winnerMessage(data.state));
      return;
    }
    if (data.result?.completed) {
      setStatus(
        `Player ${data.state.currentPlayer} scored ${data.result.completed}. Play again.`
      );
      return;
    }
    setStatus(`Player ${data.state.currentPlayer} to move.`);
  };

  const startGame = async (nextSize = size) => {
    setBusy(true);
    setBotHold(false);
    setStatus("Creating a new game...");
    try {
      const data = await fetchJson(`${API_BASE}/api/game`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ size: nextSize }),
      });
      setGameId(data.gameId);
      setState(data.state);
      setLastMove(null);
      setStatus(`Player ${data.state.currentPlayer} to move.`);
    } catch (error) {
      setStatus(error.message || "Unable to create game.");
    } finally {
      setBusy(false);
    }
  };

  const handleEdgeClick = async (x, y, direction) => {
    if (!gameId || !state || busy || state.done) {
      return;
    }
    if (getPlayerType(state.currentPlayer) !== "human") {
      return;
    }
    setBusy(true);
    try {
      const data = await fetchJson(`${API_BASE}/api/game/${gameId}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x, y, direction }),
      });
      applyMoveResponse(data, { x, y, direction });
    } catch (error) {
      setStatus(error.message || "Move rejected.");
      if (error.data?.state) {
        setState(error.data.state);
      }
    } finally {
      setBusy(false);
    }
  };

  const requestBotMove = async (botType) => {
    if (!gameId || !state || busy || state.done) {
      return;
    }
    setBusy(true);
    setStatus(`Bot (${botType}) is thinking...`);
    try {
      const data = await fetchJson(`${API_BASE}/api/game/${gameId}/bot-move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bot: botType }),
      });
      applyMoveResponse(data);
    } catch (error) {
      setStatus(error.message || "Bot move rejected.");
      setBotHold(true);
      if (error.data?.state) {
        setState(error.data.state);
      }
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!gameId || !state || busy || botHold || state.done) {
      return;
    }
    const currentType = getPlayerType(state.currentPlayer);
    if (currentType === "human") {
      return;
    }
    const timer = setTimeout(() => {
      requestBotMove(currentType);
    }, BOT_DELAY_MS);
    return () => clearTimeout(timer);
  }, [gameId, state, busy, player1Type, player2Type]);

  const boardCells = useMemo(() => {
    if (!state) {
      return [];
    }
    const cells = [];
    const gridSize = 2 * state.size + 1;
    const ownerClass = (owner) =>
      owner === 1 ? "owner-1" : owner === 2 ? "owner-2" : "";
    const isBotTurn = getPlayerType(state.currentPlayer) !== "human";

    for (let row = 0; row < gridSize; row += 1) {
      for (let col = 0; col < gridSize; col += 1) {
        const key = `${row}-${col}`;
        const isDot = row % 2 === 0 && col % 2 === 0;
        const isHorizontal = row % 2 === 0 && col % 2 === 1;
        const isVertical = row % 2 === 1 && col % 2 === 0;
        const isBox = row % 2 === 1 && col % 2 === 1;

        if (isDot) {
          cells.push(<div key={key} className="dot" />);
          continue;
        }

        if (isHorizontal) {
          const r = row / 2;
          const c = (col - 1) / 2;
          const owner = state.horizontal[r][c];
          const locked = owner !== 0 || busy || state.done || isBotTurn;
          const justDrawn =
            owner !== 0 &&
            lastMove &&
            lastMove.direction === "r" &&
            lastMove.x === c &&
            lastMove.y === r;
          cells.push(
            <button
              key={key}
              type="button"
              className={`edge horizontal ${ownerClass(owner)} ${
                locked ? "locked" : "open"
              } ${justDrawn ? "just-drawn" : ""}`}
              onClick={() => handleEdgeClick(c, r, "r")}
              disabled={locked}
              aria-label={`Horizontal edge at ${c}, ${r}`}
            >
              <span className="stroke shadow" aria-hidden="true" />
              <span className="stroke" aria-hidden="true" />
              <span className="pen-tip" aria-hidden="true" />
            </button>
          );
          continue;
        }

        if (isVertical) {
          const r = (row - 1) / 2;
          const c = col / 2;
          const owner = state.vertical[r][c];
          const locked = owner !== 0 || busy || state.done || isBotTurn;
          const justDrawn =
            owner !== 0 &&
            lastMove &&
            lastMove.direction === "d" &&
            lastMove.x === c &&
            lastMove.y === r;
          cells.push(
            <button
              key={key}
              type="button"
              className={`edge vertical ${ownerClass(owner)} ${
                locked ? "locked" : "open"
              } ${justDrawn ? "just-drawn" : ""}`}
              onClick={() => handleEdgeClick(c, r, "d")}
              disabled={locked}
              aria-label={`Vertical edge at ${c}, ${r}`}
            >
              <span className="stroke shadow" aria-hidden="true" />
              <span className="stroke" aria-hidden="true" />
              <span className="pen-tip" aria-hidden="true" />
            </button>
          );
          continue;
        }

        if (isBox) {
          const r = (row - 1) / 2;
          const c = (col - 1) / 2;
          const owner = state.boxes[r][c];
          cells.push(
            <div
              key={key}
              className={`box ${ownerClass(owner)}`}
              aria-label={`Box ${c}, ${r}`}
            />
          );
        }
      }
    }
    return cells;
  }, [state, busy, lastMove, player1Type, player2Type]);

  return (
    <div className="shell">
      <div className="glow glow-a" />
      <div className="glow glow-b" />
      <header className="hero">
        <p className="eyebrow">Dots and Boxes</p>
        <h1>Trace the edges, claim the boxes.</h1>
        <p className="lede">
          A tactile, two-player duel. Click an edge to draw a line. Close a box
          to score and keep the turn.
        </p>
      </header>

      <main className="app">
        <section className="panel" style={{ "--i": 1 }}>
          <div className="panel-head">
            <h2>Control room</h2>
            <span className={busy ? "pill busy" : "pill"}>
              {busy ? "Busy" : "Ready"}
            </span>
          </div>
          <div className="controls">
            <label className="field">
              Board size
              <select
                value={size}
                onChange={(event) => setSize(Number(event.target.value))}
                disabled={busy}
              >
                {BOARD_SIZES.map((value) => (
                  <option key={value} value={value}>
                    {value} x {value}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Player 1
              <select
                value={player1Type}
                onChange={(event) => {
                  setPlayer1Type(event.target.value);
                  setBotHold(false);
                }}
                disabled={busy}
              >
                {PLAYER_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Player 2
              <select
                value={player2Type}
                onChange={(event) => {
                  setPlayer2Type(event.target.value);
                  setBotHold(false);
                }}
                disabled={busy}
              >
                {PLAYER_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="primary"
              onClick={() => startGame(size)}
              disabled={busy}
            >
              New game
            </button>
          </div>
          <div className="status">
            <p className="status-title">Status</p>
            <p className="status-text">{status}</p>
            <p className="status-hint">
              API: {API_BASE.replace("http://", "").replace("https://", "")}
            </p>
          </div>
        </section>

        <section className="board-wrap" style={{ "--i": 2 }}>
          <div className="scoreboard">
            <div
              className={`score-card player-1 ${
                state?.currentPlayer === 1 && !state?.done ? "active" : ""
              }`}
            >
              <p>Player 1</p>
              <strong>{state?.scores?.["1"] ?? 0}</strong>
            </div>
            <div
              className={`score-card player-2 ${
                state?.currentPlayer === 2 && !state?.done ? "active" : ""
              }`}
            >
              <p>Player 2</p>
              <strong>{state?.scores?.["2"] ?? 0}</strong>
            </div>
          </div>

          <div className="board-shell">
            {state ? (
              <div
                className={`board player-${state.currentPlayer}`}
                style={{
                  "--grid": 2 * state.size + 1,
                  "--size": state.size,
                }}
              >
                {boardCells}
              </div>
            ) : (
              <div className="board-placeholder">Click New game to start.</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
