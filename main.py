from game_logic import GameState


def prompt_int(prompt: str):
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter a number, or 'q' to quit.")


def prompt_direction():
    while True:
        raw = input("direction [r=right, d=down] (q to quit): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return None
        if raw in ("r", "d"):
            return raw
        print("Please enter 'r' or 'd'.")


def main():
    size = 5
    game = GameState.new(size)
    print("Dots and Boxes")
    print("Enter coordinates for the dot and a direction.")
    print("Example: x=0, y=0, direction=r adds a top edge.")

    while not game.done:
        print()
        print(game.render_ascii())
        print(f"Score: J1 {game.scores[1]} | J2 {game.scores[2]}")
        print(f"Current player: J{game.current_player}")
        x = prompt_int("x = ? ")
        if x is None:
            print("Quit.")
            return
        y = prompt_int("y = ? ")
        if y is None:
            print("Quit.")
            return
        direction = prompt_direction()
        if direction is None:
            print("Quit.")
            return
        result = game.play_move(x, y, direction)
        if not result["valid"]:
            print(f"Invalid move: {result['message']}")
            continue
        if result["completed"]:
            print(f"Boxes completed: {result['completed']}")

    print()
    print(game.render_ascii())
    print("Game over.")
    print(f"Final score: J1 {game.scores[1]} | J2 {game.scores[2]}")
    if game.scores[1] > game.scores[2]:
        print("Winner: J1")
    elif game.scores[1] < game.scores[2]:
        print("Winner: J2")
    else:
        print("Draw.")


if __name__ == "__main__":
    main()
