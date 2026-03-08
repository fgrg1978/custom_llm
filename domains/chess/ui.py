"""
Interfaz de usuario CLI para ajedrez.
Renderizado del tablero y loop de juego.
"""

import chess


PIECE_SYMBOLS = {
    "R": "\u2656", "N": "\u2658", "B": "\u2657", "Q": "\u2655", "K": "\u2654", "P": "\u2659",
    "r": "\u265C", "n": "\u265E", "b": "\u265D", "q": "\u265B", "k": "\u265A", "p": "\u265F",
}


def render_board(board, perspective_white=True):
    """Renderiza el tablero en ASCII con piezas unicode."""
    board_str = str(board)
    rows = board_str.split("\n")

    if not perspective_white:
        rows = rows[::-1]
        rows = [row[::-1] for row in rows]

    print("\n  +---+---+---+---+---+---+---+---+")
    for i, row in enumerate(rows):
        rank = 8 - i if perspective_white else i + 1
        pieces = row.split()
        display = []
        for p in pieces:
            if p == ".":
                display.append(" ")
            else:
                display.append(PIECE_SYMBOLS.get(p, p))
        print(f"{rank} | {' | '.join(display)} |")
        print("  +---+---+---+---+---+---+---+---+")

    if perspective_white:
        print("    a   b   c   d   e   f   g   h\n")
    else:
        print("    h   g   f   e   d   c   b   a\n")


def get_human_move(board, turn_label):
    """Pide un movimiento al jugador humano."""
    while True:
        user_input = input(f"  [{turn_label}] Tu movimiento: ").strip()
        if user_input.lower() == "salir":
            return None
        try:
            move = board.parse_san(user_input)
            return move
        except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
            print(f"  Invalido. Legales: {', '.join(board.san(m) for m in board.legal_moves)}")
