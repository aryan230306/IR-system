"""Stream sources. Every source is an async iterator of `Document`.

* SyntheticNewsSource - deterministic news-wire simulator with topics, breaking-news bursts and
  syndicated near-duplicates. Carries ground-truth labels in `meta` for evaluation.
* JsonlReplaySource   - replay a JSONL file (optionally at real-time speed).
* RssSource           - poll live RSS/Atom feeds (stdlib only).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

from .models import Document

TOPICS: dict[str, dict[str, list[str]]] = {
    "technology": {
        "subj": ["Apple", "Google", "Microsoft", "Nvidia", "OpenAI", "Samsung", "Intel", "Tesla", "Meta", "Amazon"],
        "verb": ["unveils", "launches", "announces", "releases", "demonstrates", "previews"],
        "obj": ["a new smartphone", "an AI chip", "a language model", "a cloud platform", "a quantum processor",
                "a self-driving system", "a wearable device", "a cybersecurity tool", "a robotics prototype"],
        "ctx": ["software developers", "semiconductor supply chain", "machine learning researchers", "data centers",
                "mobile app ecosystem", "open source community", "chip fabrication", "privacy regulators"],
    },
    "finance": {
        "subj": ["Wall Street", "The Federal Reserve", "Goldman Sachs", "European markets", "The Nikkei", "Bitcoin",
                 "JPMorgan", "Treasury yields", "The rupee", "Oil prices"],
        "verb": ["rallies after", "slumps on", "steadies amid", "jumps following", "tumbles on", "edges higher on"],
        "obj": ["inflation data", "interest rate fears", "strong quarterly earnings", "bond market volatility",
                "central bank signals", "jobs report", "currency swings", "merger speculation"],
        "ctx": ["investors", "stock traders", "hedge funds", "retail shareholders", "equity analysts",
                "portfolio managers", "commodity futures", "bank lending"],
    },
    "sports": {
        "subj": ["Manchester United", "Real Madrid", "India", "The Lakers", "Novak Djokovic", "Ferrari",
                 "Brazil", "The Yankees", "Mumbai Indians", "Liverpool"],
        "verb": ["defeats", "edges", "thrashes", "draws with", "stuns", "outlasts"],
        "obj": ["rivals in a thrilling final", "opponents at home", "the champions on penalties",
                "challengers in extra time", "title favourites in the semifinal"],
        "ctx": ["the coach", "the striker", "tournament standings", "the goalkeeper", "fans in the stadium",
                "the league table", "the captain", "transfer window"],
    },
    "health": {
        "subj": ["Researchers", "The WHO", "A new study", "Doctors", "The FDA", "Scientists", "Hospitals",
                 "Pfizer", "A clinical trial", "Public health officials"],
        "verb": ["report", "warn of", "approve", "link", "find evidence of", "investigate"],
        "obj": ["a promising cancer therapy", "a vaccine breakthrough", "rising diabetes cases",
                "antibiotic resistance", "a new flu strain", "benefits of regular exercise", "gene editing results"],
        "ctx": ["patients", "clinical outcomes", "immune response", "drug safety", "medical researchers",
                "nutrition guidelines", "mental health services", "epidemiologists"],
    },
    "politics": {
        "subj": ["The Senate", "Parliament", "The Prime Minister", "The President", "Opposition leaders",
                 "The European Union", "The Supreme Court", "Election officials", "Diplomats", "The cabinet"],
        "verb": ["passes", "debates", "rejects", "proposes", "negotiates", "unveils"],
        "obj": ["a sweeping budget bill", "new immigration rules", "a trade agreement", "election reform",
                "climate legislation", "a defense treaty", "tax policy changes"],
        "ctx": ["voters", "lawmakers", "coalition partners", "campaign strategists", "foreign ministers",
                "the electorate", "constitutional experts", "opinion polls"],
    },
    "climate": {
        "subj": ["Scientists", "A powerful storm", "Rising seas", "The UN", "Wildfires", "A heatwave", "Glaciers",
                 "Renewable energy firms", "Coastal cities", "Farmers"],
        "verb": ["threaten", "batter", "drive", "accelerate", "warn about", "adapt to"],
        "obj": ["record global temperatures", "coastal flooding", "carbon emissions targets", "drought conditions",
                "extreme weather events", "the transition to solar power"],
        "ctx": ["greenhouse gases", "climate negotiators", "wind farms", "sea level", "biodiversity",
                "electric vehicles", "crop yields", "carbon markets"],
    },
    "science": {
        "subj": ["NASA", "ISRO", "Astronomers", "CERN physicists", "SpaceX", "A research team", "Biologists",
                 "The James Webb telescope", "Geologists", "Chemists"],
        "verb": ["discover", "launch", "observe", "map", "detect", "simulate"],
        "obj": ["a distant exoplanet", "gravitational waves", "a lunar mission", "a new particle",
                "ancient fossils", "a black hole merger", "a Mars rover sample"],
        "ctx": ["the solar system", "dark matter", "orbital mechanics", "space agencies", "telescope data",
                "particle accelerators", "planetary geology", "cosmology"],
    },
    "entertainment": {
        "subj": ["Netflix", "Disney", "Taylor Swift", "A Bollywood blockbuster", "The Oscars", "Warner Bros",
                 "Spotify", "A new sitcom", "Christopher Nolan", "The Cannes festival"],
        "verb": ["premieres", "tops", "breaks records with", "announces", "wins acclaim for", "wraps filming of"],
        "obj": ["a summer movie", "a streaming series", "a world tour", "a chart-topping album",
                "a documentary", "a superhero sequel"],
        "ctx": ["box office", "film critics", "streaming subscribers", "music fans", "award season",
                "the director", "the soundtrack", "celebrity interviews"],
    },
}

# Breaking-news events: unique vocabulary that appears suddenly (drives burst detection).
EVENTS = [
    ("climate", "earthquake", ["earthquake", "tsunami", "magnitude", "aftershock", "rescue"], "Chile"),
    ("technology", "outage", ["outage", "blackout", "servers", "ransomware", "breach"], "AWS"),
    ("sports", "worldcup", ["worldcup", "penalty", "stoppage", "hattrick", "trophy"], "Argentina"),
    ("health", "outbreak", ["outbreak", "quarantine", "pathogen", "lockdown", "contagion"], "Geneva"),
    ("finance", "crash", ["crash", "circuit", "breaker", "meltdown", "plunge"], "Tokyo"),
    ("politics", "summit", ["summit", "ceasefire", "sanctions", "envoy", "talks"], "Brussels"),
]
FIRST = ["Maria", "John", "Priya", "Chen", "Ahmed", "Sofia", "Liam", "Aarav", "Elena", "Kofi", "Yuki", "Omar", "Anna", "Rahul", "Zoe"]
LAST = ["Smith", "Patel", "Garcia", "Nguyen", "Kim", "Okafor", "Rossi", "Singh", "Muller", "Haddad", "Silva", "Ivanov", "Cohen", "Reddy"]
SOURCES = ["Reuters", "AP", "BBC", "Bloomberg", "TheHindu", "CNN", "AlJazeera", "NDTV", "Guardian", "WSJ"]


@dataclass
class SyntheticNewsSource:
    rate: float = 50.0            # docs / second (virtual if realtime=False)
    seed: int = 7
    n_docs: int | None = None     # stop after n docs (None = infinite)
    realtime: bool = True         # sleep to honour `rate`
    dup_prob: float = 0.06        # syndicated near-duplicate probability
    event_every: float = 90.0     # mean seconds (virtual) between breaking events
    event_len: float = 40.0
    start_ts: float | None = None
    id_prefix: str = ""          # keeps ids unique across restarts of a live stream

    def generate(self) -> Iterator[Document]:
        rng = random.Random(self.seed)
        ts = self.start_ts if self.start_ts is not None else time.time()
        topics = list(TOPICS)
        active: list[tuple[float, tuple]] = []
        next_event = ts + rng.expovariate(1 / self.event_every)
        recent: list[Document] = []
        i = 0
        while self.n_docs is None or i < self.n_docs:
            ts += rng.expovariate(self.rate)
            if ts >= next_event:
                active.append((ts + self.event_len, rng.choice(EVENTS)))
                next_event = ts + rng.expovariate(1 / self.event_every)
            active = [a for a in active if a[0] > ts]
            i += 1
            if recent and rng.random() < self.dup_prob:
                src = rng.choice(recent[-40:])
                yield Document(id=f"d{i}", title=src.title, source=rng.choice(SOURCES), ts=ts,
                               text=src.text + rng.choice([" (Syndicated)", " Read more.", ""]),
                               meta={**src.meta, "dup_of": src.id})
                continue
            if active and rng.random() < 0.45:                       # event coverage
                _, (topic, ename, words, place) = rng.choice(active)
                doc = self._event_doc(rng, i, ts, topic, ename, words, place)
            else:
                doc = self._topic_doc(rng, i, ts, rng.choice(topics))
            recent.append(doc)
            if len(recent) > 200:
                recent = recent[-100:]
            yield doc

    @staticmethod
    def _topic_doc(rng: random.Random, i: int, ts: float, topic: str) -> Document:
        t = TOPICS[topic]
        s, v, o = rng.choice(t["subj"]), rng.choice(t["verb"]), rng.choice(t["obj"])
        ctx = rng.sample(t["ctx"], 4)
        who = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        num = rng.choice([f"{rng.randint(2, 97)} percent", f"{rng.randint(2, 900)} million dollars",
                          f"{rng.randint(11, 9999)} people", f"{rng.randint(3, 60)} days"])
        title = f"{s} {v} {o}"
        templates = [
            f"{title}, according to people familiar with the matter. Analysts say the move will matter to "
            f"{ctx[0]} and {ctx[1]}, while {ctx[2]} could also feel the impact. {who} declined to comment.",
            f"{title}. {who}, a spokesperson, said the decision followed {num} of internal review and would "
            f"reshape {ctx[0]}. Observers expect {ctx[3]} to respond within weeks.",
            f"In a development watched closely by {ctx[1]}, {s} {v} {o}. The step involves roughly {num}, "
            f"{who} told reporters, adding that {ctx[2]} remain a priority.",
            f"{who} confirmed on Tuesday that {s} {v} {o}. Early reaction from {ctx[0]} was mixed, "
            f"with {ctx[3]} citing figures of {num} as evidence of the scale involved.",
        ]
        return Document(id=f"d{i}", title=title, text=rng.choice(templates), ts=ts,
                        source=rng.choice(SOURCES), meta={"topic": topic})

    @staticmethod
    def _event_doc(rng, i, ts, topic, ename, words, place) -> Document:
        w = rng.sample(words, 3)
        who = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        num = rng.choice([f"{rng.randint(2, 97)} percent", f"{rng.randint(2, 900)} million dollars",
                          f"{rng.randint(11, 9999)} people", f"{rng.randint(3, 60)} hours"])
        ctx = rng.sample(TOPICS[topic]["ctx"], 3)
        heads = [f"BREAKING: {w[0]} reported in {place} as {w[1]} spreads",
                 f"{place} hit by {w[0]}; {w[2]} under way",
                 f"{w[1].capitalize()} deepens after {w[0]} in {place}",
                 f"Officials in {place} respond to {w[0]}",
                 f"What we know about the {w[0]} in {place}"]
        bodies = [
            f"{who}, an official in {place}, said the {w[0]} affected about {num} and that {w[2]} efforts were "
            f"continuing. {ctx[0].capitalize()} were disrupted.",
            f"Authorities in {place} confirmed a {w[0]} on Thursday, with {w[1]} and {w[2]} dominating the "
            f"response. {ctx[1].capitalize()} and {ctx[2]} were among those affected, {who} told reporters.",
            f"Hours after the {w[0]}, {place} faced {w[1]}. Early estimates put the impact at {num}, according to "
            f"{who}. Analysts tracking {ctx[0]} expect further {w[2]}.",
            f"Witnesses described {w[1]} across {place}. {who} warned that {num} could be affected as {w[0]} "
            f"continues to unfold, with {ctx[1]} on alert.",
        ]
        return Document(id=f"d{i}", title=rng.choice(heads), text=rng.choice(bodies), ts=ts,
                        source=rng.choice(SOURCES), meta={"topic": topic, "event": ename})

    async def stream(self) -> AsyncIterator[Document]:
        for doc in self.generate():
            if self.realtime:
                await asyncio.sleep(max(0.0, 1.0 / self.rate))
                doc.ts = time.time()                                   # live: stamp at emission
                doc.id = self.id_prefix + doc.id
            else:
                await asyncio.sleep(0)
            yield doc


@dataclass
class JsonlReplaySource:
    path: str
    realtime: bool = False
    speed: float = 1.0

    async def stream(self) -> AsyncIterator[Document]:
        prev = None
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = Document.from_dict(json.loads(line))
                if self.realtime and prev is not None:
                    await asyncio.sleep(max(0.0, (doc.ts - prev) / self.speed))
                else:
                    await asyncio.sleep(0)
                prev = doc.ts
                yield doc


@dataclass
class RssSource:
    feeds: list[str]
    poll_seconds: float = 60.0
    _seen: set = None

    def _fetch(self, url: str) -> list[Document]:
        req = urllib.request.Request(url, headers={"User-Agent": "StreamIR/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            root = ET.fromstring(r.read())
        out = []
        for it in list(root.iter("item")) + list(root.iter("{http://www.w3.org/2005/Atom}entry")):
            def g(tag):
                e = it.find(tag) if it.find(tag) is not None else it.find("{http://www.w3.org/2005/Atom}" + tag)
                return (e.text or "").strip() if e is not None else ""
            title, desc = g("title"), g("description") or g("summary")
            key = g("guid") or g("id") or g("link") or title
            out.append(Document(id=hashlib.sha1(key.encode()).hexdigest()[:16], title=title, text=desc or title,
                                ts=time.time(), source=url.split("/")[2], meta={"link": g("link")}))
        return out

    async def stream(self) -> AsyncIterator[Document]:
        self._seen = set()
        while True:
            for url in self.feeds:
                try:
                    docs = await asyncio.to_thread(self._fetch, url)
                except Exception:                                     # network errors must not kill the stream
                    continue
                for d in docs:
                    if d.id not in self._seen:
                        self._seen.add(d.id)
                        yield d
            await asyncio.sleep(self.poll_seconds)
