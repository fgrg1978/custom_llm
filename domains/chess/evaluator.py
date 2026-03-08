"""
Evaluadores de posiciones de ajedrez para RLHF.
Stockfish (preciso) o heuristico (sin dependencias externas).
"""

import os
import chess
import chess.engine


STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
]


def find_stockfish():
    """Busca el binario de Stockfish."""
    for path in STOCKFISH_PATHS:
        if os.path.exists(path):
            return path
    return None


class StockfishEvaluator:
    """Evalua posiciones usando Stockfish."""

    def __init__(self, depth=12, time_limit=0.1):
        path = find_stockfish()
        if not path:
            raise FileNotFoundError(
                "Stockfish no encontrado. Instalalo con: brew install stockfish"
            )
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth
        self.time_limit = time_limit

    def evaluate(self, board):
        result = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth, time=self.time_limit),
        )
        score = result["score"].relative
        if score.is_mate():
            mate_in = score.mate()
            return 10000 if mate_in > 0 else -10000
        return score.score()

    def get_reward(self, board_before, board_after):
        score_before = self.evaluate(board_before)
        score_after = -self.evaluate(board_after)
        diff = (score_after - score_before) / 100.0
        return max(-5.0, min(5.0, diff))

    def close(self):
        self.engine.quit()


class HeuristicEvaluator:
    """Evalua posiciones sin motor externo, usando reglas basicas."""

    PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
    }

    def evaluate(self, board):
        if board.is_checkmate():
            return -10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type] * 100
                if piece.color == board.turn:
                    score += value
                else:
                    score -= value

        score += len(list(board.legal_moves)) * 5

        center = [chess.E4, chess.D4, chess.E5, chess.D5]
        for sq in center:
            if board.is_attacked_by(board.turn, sq):
                score += 10

        return score

    def get_reward(self, board_before, board_after):
        score_before = self.evaluate(board_before)
        score_after = -self.evaluate(board_after)
        diff = (score_after - score_before) / 100.0
        return max(-5.0, min(5.0, diff))

    def close(self):
        pass
