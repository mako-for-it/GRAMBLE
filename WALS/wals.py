from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class WALSProfile:
    # store your dataset’s codes verbatim
    f81A: str | None = None  # e.g. "81A-1"
    f84A: str | None = None
    f85A: str | None = None
    f86A: str | None = None
    f87A: str | None = None
    f88A: str | None = None
    f89A: str | None = None

Token = Tuple[str, str]

wals_profiles = {
    "spa": WALSProfile(f81A = "81A-2", f84A= "84A-1", f85A= "85A-2", f86A= "86A-2", f87A="87A-2", f88A= "88A-1", f89A= "89A-1"),
    "dut": WALSProfile(f81A = "81A-7", f84A= "84A-6", f85A= "85A-2", f86A= "86A-2", f87A="87A-1", f88A= "88A-1", f89A= "89A-1"),
    "afr": WALSProfile(f81A = None, f84A= None, f85A= None, f86A= None, f87A= None, f88A= None, f89A= None),
    "xho": WALSProfile(f81A = "81A-2", f84A= None, f85A= "85A-2", f86A= "86A-2", f87A="87A-2", f88A= "88A-6", f89A= "89A-2"),
    "ass": WALSProfile(f81A = "81A-1", f84A= None, f85A= None, f86A= "86A-1", f87A="87A-1", f88A= "88A-1", f89A= "89A-3"),
    "jap": WALSProfile(f81A = "81A-1", f84A= "84A-3", f85A= "85A-1", f86A= "86A-1", f87A="87A-1", f88A= "88A-1", f89A= "89A-1"),
}

#Order of Subject, Object and verb
ORDER_81A = {
    "81A-1": ["S", "O", "V"],
    "81A-2": ["S", "V", "O"],
    "81A-3": ["V", "S", "O"],
    "81A-4": ["V", "O", "S"],
    "81A-5": ["O", "V", "S"],
    "81A-6": ["O", "S", "V"],
}

#Order of Object, Oblique, and verb
ORDER_84A = {
    "84A-1": ["V", "O", "X"],
    "84A-2": ["X", "V", "O"],
    "84A-3": ["X", "O", "V"],
    "84A-4": ["O", "X", "V"],
    "84A-5": ["O", "V", "X"],
}

#Order of Adposition and Noun Phrase
ORDER_85A = {
    "85A-1": ["NP", "prepositional phrase"],
    "85A-2": ["prepositional phrase", "NP"],
}

#Order of Genitive and Noun
ORDER_86A = {
    "86A-1": ["Gen", "N"],
    "86A-2": ["N", "Gen"],
}

#Order of Adjective and Noun
ORDER_87A = {
    "87A-1": ["Adj", "N"],
    "87A-2": ["N", "Adj"],
}

#Order of Demonstrative and Noun
ORDER_88A = {
    "88A-1": ["Dem", "N"],
    "88A-2": ["N", "Dem"]
}

#Order of Numeral and Noun
ORDER_89A = {
    "89A-1": ["Num", "N"],
    "89A-2": ["N", "Num"],
}

def is_gen_token(tok: Token) -> bool:
    w, pos = tok
    return pos == "GEN" or w.endswith("'s") or w.endswith("’s")

def chunk_np(tokens: List[Token]) -> List[Tuple[str, List[Token]]]:
    """
    NP chunking:
      - pronoun alone is an NP (important!)
      - otherwise: (determiner/DEM)* (numeral)* (adjective)* (GEN)* (noun/proper noun) as NP
    """
    chunks: List[Tuple[str, List[Token]]] = []
    i = 0

    while i < len(tokens):
        w, p = tokens[i]

        # ✅ PRONOUN is a complete NP by itself (fixes your example)
        if p == "pronoun":
            chunks.append(("NP", [tokens[i]]))
            i += 1
            continue

        # Start NP only if it looks like an NP start
        can_start = (p in {"determiner", "DEM", "numeral", "adjective", "noun", "proper noun"} or is_gen_token(tokens[i]))
        if not can_start:
            chunks.append(("OTHER", [tokens[i]]))
            i += 1
            continue

        start = i
        j = i

        # (determiner/DEM)*
        while j < len(tokens) and tokens[j][1] in {"determiner", "DEM"}:
            j += 1
        # numeral*
        while j < len(tokens) and tokens[j][1] == "numeral":
            j += 1
        # adjective*
        while j < len(tokens) and tokens[j][1] == "adjective":
            j += 1
        # GEN*
        while j < len(tokens) and (tokens[j][1] == "GEN" or is_gen_token(tokens[j])):
            j += 1

        # Require a head noun/proper noun (for this NP pattern)
        if j < len(tokens) and tokens[j][1] in {"noun", "proper noun"}:
            j += 1
            chunks.append(("NP", tokens[start:j]))
            i = j
            continue

        # fallback: not a valid NP chunk
        chunks.append(("OTHER", [tokens[i]]))
        i += 1

    return chunks


def flatten(chunks: List[Tuple[str, List[Token]]]) -> List[Token]:
    out: List[Token] = []
    for _, ch in chunks:
        out.extend(ch)
    return out


# ---------------------------
# 86–89: NP internal order
# ---------------------------

def reorder_np(np_tokens: List[Token], profile: WALSProfile) -> List[Token]:
    # We treat determiner as "Dem" unless you have DEM tag.
    dem: List[Token] = []
    num: List[Token] = []
    adj: List[Token] = []
    gen: List[Token] = []
    head: List[Token] = []
    other: List[Token] = []

    for tok in np_tokens:
        w, pos = tok
        if pos == "DEM":
            dem.append(tok)
        elif pos == "determiner":
            dem.append(tok)  # approximating 88A using determiner
        elif pos == "numeral":
            num.append(tok)
        elif pos == "adjective":
            adj.append(tok)
        elif pos == "GEN" or is_gen_token(tok):
            gen.append(tok)
        elif pos in {"noun", "proper noun", "pronoun"}:
            head.append(tok)
        else:
            other.append(tok)

    if not head:
        return np_tokens

    # keep a single head to avoid weirdness
    head = [head[-1]]

    dem_order = ORDER_88A.get(profile.f88A, ["Dem", "N"])
    num_order = ORDER_89A.get(profile.f89A, ["Num", "N"])
    adj_order = ORDER_87A.get(profile.f87A, ["Adj", "N"])
    gen_order = ORDER_86A.get(profile.f86A, ["Gen", "N"])

    dem_before = (dem_order == ["Dem", "N"])
    num_before = (num_order == ["Num", "N"])
    adj_before = (adj_order == ["Adj", "N"])
    gen_before = (gen_order == ["Gen", "N"])

    left: List[Token] = []
    right: List[Token] = []

    (left if dem_before else right).extend(dem)
    (left if num_before else right).extend(num)
    (left if adj_before else right).extend(adj)
    (left if gen_before else right).extend(gen)

    return left + head + right + other

def apply_np_rules(tokens: List[Token], profile: WALSProfile) -> List[Token]:
    chunks = chunk_np(tokens)
    out_chunks: List[Tuple[str, List[Token]]] = []
    for typ, ch in chunks:
        if typ == "NP":
            out_chunks.append(("NP", reorder_np(ch, profile)))
        else:
            out_chunks.append((typ, ch))
    return flatten(out_chunks)


# ---------------------------
# 81A: clause S/O/V reorder
# ---------------------------

def is_single_pos_chunk(chunks, i: int, pos: str) -> bool:
    return (
        0 <= i < len(chunks)
        and chunks[i][0] == "OTHER"
        and len(chunks[i][1]) == 1
        and chunks[i][1][0][1] == pos
    )

def find_oblique_X(chunks: List[Tuple[str, List[Token]]], start_idx: int):
    """
    Finds the first oblique phrase X after start_idx.
    We treat X as either:
      - prepositional phrase + NP
      - NP + prepositional phrase
    Returns (X_tokens, used_chunk_indices). If not found: ([], set()).
    """
    i = start_idx
    while i < len(chunks) - 1:
        # prepositional phrase + NP
        if is_single_pos_chunk(chunks, i, "prepositional phrase") and chunks[i + 1][0] == "NP":
            X = chunks[i][1] + chunks[i + 1][1]
            return X, {i, i + 1}

        # NP + prepositional phrase
        if chunks[i][0] == "NP" and is_single_pos_chunk(chunks, i + 1, "prepositional phrase"):
            X = chunks[i][1] + chunks[i + 1][1]
            return X, {i, i + 1}

        i += 1

    return [], set()

def reorder_clause_by_81A(tokens: List[Token], profile: WALSProfile) -> List[Token]:
    order81 = ORDER_81A.get(profile.f81A)
    if not order81:
        return tokens

    chunks = chunk_np(tokens)

    # find first verb token chunk
    v_idx = next((i for i, (typ, ch) in enumerate(chunks)
                  if typ == "OTHER" and len(ch) == 1 and ch[0][1] == "verb"), None)
    if v_idx is None:
        return tokens

    # subject: first NP before V
    s_idx = next((i for i, (typ, _) in enumerate(chunks[:v_idx]) if typ == "NP"), None)
    # object: first NP after V
    o_idx = next((i for i, (typ, _) in enumerate(chunks[v_idx+1:], start=v_idx+1) if typ == "NP"), None)

    if s_idx is None or o_idx is None:
        return tokens

    S = chunks[s_idx][1]
    V = chunks[v_idx][1]
    O = chunks[o_idx][1]

    used = {s_idx, v_idx, o_idx}

    # ---- 84A: reorder O / X / V (only if we can find X)
    order84 = ORDER_84A.get(profile.f84A)
    X: List[Token] = []
    used_x: set[int] = set()

    if order84:
        X, used_x = find_oblique_X(chunks, start_idx=o_idx + 1)
        if X:
            used |= used_x

    # Build the clause backbone
    def ovx_backbone() -> List[Token]:
        # If 84A unavailable or X not found, fall back to basic O,V order from 81A
        if not order84 or not X:
            return O + V  # minimal; S positioning handled outside

        blocks84 = {"O": O, "V": V, "X": X}
        out84: List[Token] = []
        for k in order84:
            out84.extend(blocks84[k])
        return out84

    backbone = ovx_backbone()

    # Now place S relative to the backbone using 81A.
    # We interpret 81A as controlling relative positions of S / (O,V) broadly.
    # If 81A says S-first => S + backbone
    # If 81A says verb-initial etc, we approximate using S position vs backbone.
    out: List[Token] = []

    if order81[0] == "S":
        out.extend(S)
        out.extend(backbone)
    elif order81[0] == "V":
        # V-initial: put V first, then S, then O/X remainder (approx)
        # We'll use the backbone but rotate to start with V if possible
        if order84 and X:
            # backbone already has V somewhere; enforce starting with V
            out.extend(V)
            out.extend(S)
            # then add O+X in their 84A relative order excluding V
            for k in order84:
                if k != "V":
                    out.extend({"O": O, "X": X}[k])
        else:
            out.extend(V)
            out.extend(S)
            out.extend(O)
    elif order81[0] == "O":
        # O-initial: O + S + V (rare; approx)
        out.extend(O)
        out.extend(S)
        # then add X and V
        if order84 and X:
            # respect 84A for (X,V) order after O
            for k in order84:
                if k == "O":
                    continue
                out.extend({"V": V, "X": X}[k])
        else:
            out.extend(V)
    else:
        # unknown; do nothing
        return tokens

    # keep remaining chunks in original order (don't drop words)
    for i, (_, ch) in enumerate(chunks):
        if i not in used:
            out.extend(ch)

    return out

# ---------------------------
# 85A: prepositional phrase <-> NP order
# ---------------------------

def apply_85A_adposition(tokens: List[Token], profile: WALSProfile) -> List[Token]:
    rule = ORDER_85A.get(profile.f85A)
    if not rule:
        return tokens

    chunks = chunk_np(tokens)
    out: List[Token] = []
    i = 0
    while i < len(chunks):
        typ, ch = chunks[i]

        # Pattern: prepositional phrase + NP
        if typ == "OTHER" and len(ch) == 1 and ch[0][1] == "prepositional phrase":
            if i + 1 < len(chunks) and chunks[i + 1][0] == "NP":
                adp = ch
                np_ = chunks[i + 1][1]
                out.extend(adp + np_ if rule == ["prepositional phrase", "NP"] else np_ + adp)
                i += 2
                continue

        # Pattern: NP + prepositional phrase
        if typ == "NP":
            if i + 1 < len(chunks) and chunks[i + 1][0] == "OTHER" and len(chunks[i+1][1]) == 1 and chunks[i+1][1][0][1] == "prepositional phrase":
                np_ = ch
                adp = chunks[i + 1][1]
                out.extend(adp + np_ if rule == ["prepositional phrase", "NP"] else np_ + adp)
                i += 2
                continue

        out.extend(ch)
        i += 1

    return out


# ---------------------------
# Full reorder pipeline
# ---------------------------

def reorder_sentence(tokens: List[Token], target_profile: WALSProfile) -> List[Token]:
    # 1) NP-internal order first (86–89)
    tokens = apply_np_rules(tokens, target_profile)

    # 2) Clause order (81A)
    tokens = reorder_clause_by_81A(tokens, target_profile)

    # 3) prepositional phrase/NP order (85A)
    tokens = apply_85A_adposition(tokens, target_profile)

    return tokens

tokens = [
  ("I","pronoun"),
  ("gave","verb"),
  ("her","pronoun"),
  ("the","determiner"),
  ("book","noun"),
]

if __name__ == "__main__":
    lang = input("Target language code (spa/dut/afr/xho/ass/jap): ").strip()
    prof = wals_profiles.get(lang)
    if prof is None:
        raise ValueError(f"Unknown target language: {lang}")

    out = reorder_sentence(tokens, prof)

    print("IN :", " ".join(w for w, _ in tokens))
    print("OUT:", " ".join(w for w, _ in out))
    print("PROFILE USED:", prof)
