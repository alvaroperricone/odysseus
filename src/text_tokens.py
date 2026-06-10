"""text_tokens.py: tiny, dependency-free content tokenizer.

Shared stopword tokenizer used by retrieval/ranking code. Kept in its own module
(imports only `re`) so lightweight consumers like the knowledge-only reasoner
loop can use it without dragging in the web-search / chat stack.

Mirrors the tokenizer historically defined in src/chat_processor.py.
"""

import re

STOPWORDS = frozenset(
    "a an the is am are was were be been being have has had do does did "
    "will would shall should can could may might must need ought dare "
    "i me my mine we us our ours you your yours he him his she her hers "
    "it its they them their theirs this that these those "
    "and but or nor not no so if then else than too also very "
    "in on at to for of by with from up out about into over after "
    "what when where which who whom how why all each every some any "
    "just very really actually like well also still already even "
    "oh ok okay yes yeah hey hi hello thanks thank please sorry "
    "much more most own other another such only same here there "
    "because while during before until since through between both "
    "few many several some none nothing something anything everything "
    "get got make made go going went been come came take took "
    "know think want let say tell give see look find way thing "
    "don doesn didn won wouldn couldn shouldn wasn weren isn aren haven hasn "
    "don't doesn't didn't won't wouldn't couldn't shouldn't "
    "it's i'm i've i'll i'd you're you've you'll he's she's we're we've they're they've "
    "that's there's here's what's who's how's let's can't "
    # Italian (italiano)
    "il lo la i gli le un uno una di a da in con su per tra fra ed od "
    "che chi cui non è sono ho hai ha abbiamo avete hanno mi ti si ci vi ne "
    "come dove quando perche perché piu più meno molto poco anche ancora gia già "
    "sempre mai qui qua questo questa questi queste quello quella quelli quelle "
    "del dello della dei degli delle al allo alla ai agli alle dal dallo dalla "
    "nel nello nella negli nelle sul sullo sulla col coi "
    "mio mia miei mie tuo tua tuoi tue suo sua nostro nostra vostro loro "
    "essere fare cosa quale quali ogni tutto tutta tutti tutte cosi così "
    "bene grazie ciao si no piu meno se ma o oppure quindi pero però allora".split()
)


def content_tokens(text: str) -> list:
    """Meaningful content words: no stopwords, min 3 chars, lowercased. Keeps
    accented Latin letters so Italian words aren't truncated (perché, città)."""
    words = re.findall(r'[a-zà-ÿ0-9]+(?:[-_][a-zà-ÿ0-9]+)*', (text or "").lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]
