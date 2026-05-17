"""
Descarga y prepara datos de partidas de ajedrez desde Lichess.

Los archivos pesados (.zst, .pgn) se guardan en /tmp/ para no contaminar
el proyecto. Solo vocab.json y sequences.pt se guardan en data/.

Preparacion PARALELA (el cuello de botella era el parsing PGN single-thread):
  1. Descomprime el .zst a un .pgn en disco (seekable).
  2. Parte el .pgn en N trozos por offsets de bytes, alineados a "[Event ".
  3. N workers en paralelo: cada uno usa read_headers() para filtrar por ELO
     barato (salta el movetext) y solo hace read_game() completo si pasa.
  4. Merge de los move-sets -> vocab, concat de raw_games -> tokeniza -> guarda.

Speedup tipico: ~20-30x vs el escaneo single-thread.
"""

import os
import io
import subprocess
import multiprocessing as mp

import chess.pgn
import zstandard
from tqdm import tqdm

from domains.chess.tokenizer import (
    build_vocab_and_parse, elo_bucket_token, _read_elos,
    RESULT_MAP, RESULT_TOKENS, ELO_BUCKET_TOKENS,
)
from core.dataset import save_vocab, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json")
DATA_FILE = os.path.join(DATA_DIR, "sequences.pt")

DEFAULT_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2022-11.pgn.zst"


def _zst_path_for_url(url):
    """Devuelve la ruta en /tmp/ para el .zst de una URL dada."""
    filename = url.split("/")[-1]
    return os.path.join("/tmp", filename)


def _pgn_path_for_zst(zst_file):
    """Ruta del .pgn descomprimido (en /tmp/, mismo nombre sin .zst)."""
    base = os.path.basename(zst_file)
    if base.endswith(".zst"):
        base = base[:-4]
    return os.path.join("/tmp", base)


def _remote_size(url):
    """Tamano del archivo remoto via HTTP HEAD. None si no se puede obtener."""
    try:
        out = subprocess.run(
            ["/usr/bin/curl", "-sIL", url],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in out.splitlines():
            if line.lower().startswith("content-length:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def download_zst(url):
    """Descarga el .zst a /tmp/. Verifica integridad y reanuda descargas parciales.

    Casos:
      - No existe           -> descarga completa
      - Existe e incompleto -> reanuda con curl -C - (compara con tamano remoto)
      - Existe y completo   -> no descarga nada
    """
    zst_file = _zst_path_for_url(url)
    expected = _remote_size(url)

    if os.path.exists(zst_file):
        local = os.path.getsize(zst_file)
        if expected is not None and local == expected:
            print(f"Archivo ya existe y esta completo: {zst_file} ({local/1e9:.1f} GB)")
            return zst_file
        elif expected is not None and local < expected:
            print(f"Archivo incompleto: {local/1e9:.1f}/{expected/1e9:.1f} GB -> reanudando descarga")
        elif expected is None:
            print(f"Archivo existe ({local/1e9:.1f} GB) pero no se pudo verificar tamano remoto.")
            print(f"  Reanudando por seguridad (curl -C - no hace nada si ya esta completo)")
    else:
        print(f"Descargando: {url}")
        print(f"Destino:     {zst_file}")

    subprocess.run(
        ["/usr/bin/curl", "-L", "-C", "-", "--progress-bar", "-o", zst_file, url],
        check=True,
    )
    size_gb = os.path.getsize(zst_file) / 1e9
    print(f"Descarga lista: {zst_file} ({size_gb:.1f} GB)")
    return zst_file


def open_pgn_stream(zst_file):
    """Abre el .zst como stream de texto sin descomprimir a disco (uso single-thread)."""
    dctx = zstandard.ZstdDecompressor()
    fh = open(zst_file, "rb")
    stream = dctx.stream_reader(fh)
    return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")


def decompress_zst(zst_file, pgn_file=None, max_bytes=None):
    """Descomprime el .zst a un .pgn en disco (seekable, necesario para parsing paralelo).

    max_bytes: si se da, para tras escribir ~max_bytes (suficiente para N partidas).
    Si el .pgn ya existe con tamano suficiente, no hace nada.
    """
    if pgn_file is None:
        pgn_file = _pgn_path_for_zst(zst_file)

    if os.path.exists(pgn_file):
        size = os.path.getsize(pgn_file)
        if max_bytes is None or size >= max_bytes:
            print(f"PGN ya descomprimido: {pgn_file} ({size/1e9:.1f} GB)")
            return pgn_file

    print(f"Descomprimiendo {zst_file} -> {pgn_file}...")
    dctx = zstandard.ZstdDecompressor()
    written = 0
    chunk_size = 64 * 1024 * 1024  # 64 MB
    with open(zst_file, "rb") as ifh, open(pgn_file, "wb") as ofh:
        reader = dctx.stream_reader(ifh)
        pbar = tqdm(desc="Descomprimiendo", unit="GB", unit_scale=False)
        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break
            ofh.write(chunk)
            written += len(chunk)
            pbar.n = round(written / 1e9, 1)
            pbar.refresh()
            if max_bytes is not None and written >= max_bytes:
                break
        pbar.close()
    print(f"PGN listo: {pgn_file} ({written/1e9:.1f} GB)")
    return pgn_file


def _find_chunk_starts(pgn_file, n_chunks):
    """Devuelve n offsets de bytes, cada uno al inicio de una partida ('[Event ').

    Los workers arrancan en estos offsets. Como cada worker para al llegar a su
    cuota (consume ~MB de un trozo de ~GB), nunca se solapan: no hacen falta
    boundaries de fin.
    """
    total = os.path.getsize(pgn_file)
    approx = total // n_chunks
    starts = [0]
    with open(pgn_file, "rb") as f:
        for i in range(1, n_chunks):
            f.seek(i * approx)
            f.readline()  # descarta linea parcial
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith(b"[Event "):
                    starts.append(pos)
                    break
    return sorted(set(starts))


def _parse_chunk(args):
    """Worker: lee partidas desde el offset `start` del .pgn hasta llenar `quota`.

    Abre en binario, hace seek al byte de inicio (siempre el comienzo de un
    '[Event '), y envuelve en TextIOWrapper para que python-chess lo lea como
    texto. Lectura puramente secuencial: sin tell()/seek() fragiles.

    Devuelve (move_set, raw_games) con raw_games = [(elo_tok, [san], result), ...].
    """
    pgn_file, start, min_elo, min_moves, quota = args
    moves_set = set()
    raw_games = []

    raw_fh = open(pgn_file, "rb")
    raw_fh.seek(start)
    f = io.TextIOWrapper(raw_fh, encoding="utf-8", errors="replace")
    try:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            white_elo, black_elo = _read_elos(game)
            if white_elo is None or min(white_elo, black_elo) < min_elo:
                continue

            result = game.headers.get("Result", "*")
            if result not in RESULT_MAP:
                continue

            board = game.board()
            san_moves = []
            for move in game.mainline_moves():
                san_moves.append(board.san(move))
                board.push(move)

            if len(san_moves) < min_moves:
                continue

            moves_set.update(san_moves)
            raw_games.append((elo_bucket_token(white_elo, black_elo), san_moves, result))

            if quota and len(raw_games) >= quota:
                break
    finally:
        f.close()

    return moves_set, raw_games


def prepare_parallel(max_games, min_elo, zst_file, n_workers=None, min_moves=10):
    """Preparacion paralela: descomprime, parte, parsea en N workers, merge."""
    import torch

    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 6)

    # Descomprimir solo lo necesario: ~25 KB/partida-objetivo cubre el pass-rate
    # del filtro ELO + tamano de partida + margen. Para 1M partidas -> ~25 GB.
    max_bytes = max_games * 25_000 if max_games else None
    pgn_file = decompress_zst(zst_file, max_bytes=max_bytes)

    print(f"\nParticionando {pgn_file} en {n_workers} trozos...")
    starts = _find_chunk_starts(pgn_file, n_workers)
    print(f"  {len(starts)} trozos")

    # Cuota por worker con margen (1.4x) por si el pass-rate es desigual entre trozos
    per_worker_quota = None
    if max_games:
        per_worker_quota = int(max_games / len(starts) * 1.4) + 1

    args_list = [
        (pgn_file, start, min_elo, min_moves, per_worker_quota)
        for start in starts
    ]

    print(f"Parseando en paralelo (cuota/worker: {per_worker_quota})...")
    move_set = set()
    raw_games = []
    with mp.Pool(n_workers) as pool:
        for ms, rg in tqdm(pool.imap_unordered(_parse_chunk, args_list),
                           total=len(args_list), desc="Workers"):
            move_set |= ms
            raw_games.extend(rg)

    if not raw_games:
        return None, None, []

    # Truncar al objetivo exacto
    if max_games and len(raw_games) > max_games:
        raw_games = raw_games[:max_games]

    # Vocab del conjunto recolectado
    special = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + RESULT_TOKENS + ELO_BUCKET_TOKENS
    vocab = special + sorted(move_set)
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}

    # Tokenizacion en memoria
    bos, eos = token_to_id[BOS_TOKEN], token_to_id[EOS_TOKEN]
    sequences = []
    for elo_tok, san_moves, result in raw_games:
        seq = (
            [bos, token_to_id[elo_tok]]
            + [token_to_id[s] for s in san_moves]
            + [token_to_id[RESULT_MAP[result]], eos]
        )
        sequences.append(seq)

    print(f"Vocabulario: {len(token_to_id)} tokens")
    print(f"Partidas validas: {len(sequences)}")
    return token_to_id, id_to_token, sequences


def prepare(max_games=50000, min_elo=1600, url=DEFAULT_URL, parallel=True, n_workers=None):
    """Pipeline completo de preparacion de datos.

    parallel=True (default): descomprime a disco y parsea en N workers (~20-30x).
    parallel=False: escaneo single-thread en streaming desde el .zst (sin .pgn en disco).
    """
    import torch

    os.makedirs(DATA_DIR, exist_ok=True)
    zst_file = download_zst(url)

    if parallel:
        token_to_id, id_to_token, sequences = prepare_parallel(
            max_games, min_elo, zst_file, n_workers=n_workers
        )
    else:
        print(f"\nConstruyendo vocabulario + parseando (single-thread, max {max_games}, min ELO {min_elo})...")
        with open_pgn_stream(zst_file) as stream:
            token_to_id, id_to_token, sequences = build_vocab_and_parse(
                stream, max_games=max_games, min_elo=min_elo
            )

    if not sequences:
        print(f"\nERROR: ninguna partida paso el filtro de ELO {min_elo}.")
        print(f"  Prueba con --min-elo mas bajo o usa una dump diferente.")
        return
    if len(sequences) < 1000:
        print(f"\nAVISO: solo {len(sequences)} partidas pasaron el filtro.")

    save_vocab(token_to_id, VOCAB_FILE)
    torch.save(sequences, DATA_FILE)
    print(f"Secuencias guardadas: {len(sequences)}")

    lengths = [len(s) for s in sequences]
    print(f"\nEstadisticas:")
    print(f"  Partidas:       {len(sequences)}")
    print(f"  Largo promedio: {sum(lengths)/len(lengths):.0f} tokens")
    print(f"  Largo maximo:   {max(lengths)}")
    print(f"  Largo minimo:   {min(lengths)}")
    print(f"\nArchivos del proyecto:")
    print(f"  {VOCAB_FILE}")
    print(f"  {DATA_FILE}")
