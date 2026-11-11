"""Knowledge system for the credit-union snapshot hackathon.

The task: answer members' questions from five downloads of a website that contradicts itself (old
rates on calculator pages, old figures in old press releases, promotions that ended but are still
advertised), with the value that is true *now* - or on an `as_of` date - and list the contradictions.

How it decides what is true:

1. Every snapshot is kept whole, so an `as_of` question is answered from the website as it stood then.
2. Pages are ranked by trust using general URL conventions: rate pages and product pages are
   maintained; campaign pages, news, blog posts, calculators and uploaded files are not.
3. Rates come only from the official rate sheet, parsed into structured rows.  The question's
   qualifiers (term, model year, balance, loan amount, jumbo, business, IRA, refinance) select rows;
   if nothing on the sheet matches, the answer is an honest "the website doesn't list that".
4. Answers contain current values only.  History goes in `conflict_note`, never in the text.
5. Figures the site restates in many press releases (branch count, members, assets) are learned
   from the text itself; the newest release wins.
6. Everything else is answered by a BM25-style passage search that prefers trusted pages and newer
   releases, and declines when the best answer does not cover the question.
7. `conflicts()` reports stale copies of rate tables, changes to the rate sheet over time, offers
   that ended but are still advertised, restated figures that changed, and differing branch hours.

Nothing is specific to one website: the organisation's name, its abbreviations, its branch locator
and its page sections are all learned from the snapshots.  Every tuning number is in the Settings
block below.  Standard library only; no network calls and no API keys.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from datetime import date
from math import log
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source # noqa: E402


# ==============================================================================================
# Settings
#
# Every number that tunes ranking and matching lives here, with the reason it exists.  They are
# deliberately round: each was set by asking which page a careful human reader would pick, then
# checked against the practice questions and several hundred reworded and "as of" questions.
# ==============================================================================================

# Page trust (see PageInfo): the rate sheet and product pages are maintained; campaign pages are
# sometimes forgotten; news, blog posts and calculators keep whatever was true when they were written.
TIER_WEIGHT = {0: 1.3, 1: 1.3, 2: 1.1, 3: 1.0}

# Passage ranking (see Site.search) - a BM25-style score built in layers.
SECTION_WEIGHT = 0.9          # a question word in the passage's sub-heading counts almost as much as in its text
TITLE_ONLY_WEIGHT = 0.75      # word only in the page title: the page is about it, this passage may not be
TITLE_TOO_WEIGHT = 0.35       # small extra when the word is in both the title and the passage
REPEAT_WEIGHT = 0.45          # a word said several times marks the passage's subject (log-scaled)
LENGTH_BASE, LENGTH_DAMPING = 0.7, 0.3    # long passages match words by accident, so damp them
MISSING_NAME_FACTOR = 0.35    # the passage does not mention a name the question asked about
LISTING_FACTOR = 0.5          # index pages, duplicate URLs and older editions of a re-issued release
FILE_FACTOR = 0.8             # calculators and uploaded files
RECENCY_FLOOR, RECENCY_SPAN = 0.85, 0.3   # oldest press release x0.85 ... newest x1.15
SHORT_PASSAGE_FACTOR = 0.6    # under 25 characters is usually a label or a button
PAGE_COVERAGE_BONUS = 0.4     # credit for question words found in other passages of the same page
PHRASE_BONUS = 0.3            # question words found side by side, as in the question

# Choosing and trusting a search answer (see KnowledgeSystem._answer_search).
TEASER_FACTOR = 0.3           # a "related article" teaser inside a release describes a different release
DATED_PASSAGE_FACTOR = 1.35   # "when ...?" questions prefer passages that contain a date
ELIGIBILITY_FACTOR = 2.0      # "who can join?" prefers the section that is about eligibility
HOW_TO_ARCHIVE_FACTOR = 0.75  # procedures live on service pages, not in blog posts
NEWEST_RELEASE_WINDOW = 0.8   # releases scoring within 20% of the best: the newest one describes the present
MIN_COVERAGE = 0.3            # share of the question (weighted by word rarity) the answer must cover
MIN_COVERAGE_SHORT = 0.67     # ... for questions of one or two words, which have less to go on

# Rates (see KnowledgeSystem._match_rows).
RATE_MATCH_WINDOW = 0.72      # rows scoring within 28% of the best match describe the same product

# Repeated facts in press releases (see Site.repeated_facts): a sentence pattern counts as restated
# boilerplate when it appears in at least this many dated articles, or this share of them.
REPEATED_MIN_PAGES = 5
REPEATED_SHARE = 0.03


# ==============================================================================================
# Text normalisation
#
# Questions and pages are rewritten into one shared vocabulary, so different phrasings meet the
# same words.  The rules are general banking and English vocabulary, not names from one website.
# ==============================================================================================

_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
NUMBER_WORD = re.compile(r"\b(?:(%s)(?:[\s-](%s))?|(%s))\b(?=[\s-]*(?:months?|years?|yrs?|mos?)\b)"
                         % ("|".join(_TENS), "|".join(_UNITS[1:10]), "|".join(_UNITS[1:])))


def _number_word(match):
    """Number words before a term: "seventeen-month" -> "17", "twenty five year" -> "25"."""
    if match.group(1):
        return str(_TENS[match.group(1)] + (_UNITS.index(match.group(2)) if match.group(2) else 0))
    return str(_UNITS.index(match.group(3)))


PHRASES = [
    (NUMBER_WORD, _number_word),
    (re.compile(r"(?<=[a-z])-(?=[a-z])"), " "),
    (re.compile(r"\brefinanc\w*|\brefi\b"), "refinance"),
    (re.compile(r"\bcertificates? of deposit\b"), "certificate"),
    (re.compile(r"\bshare certificates?\b"), "certificate"),
    (re.compile(r"\bcds?\b"), "certificate"),
    (re.compile(r"\brecreational vehicles?\b|\bcampers?\b|\bmotor ?homes?\b"), "rv"),
    (re.compile(r"\b(?:cars?|vehicles?|automobiles?|trucks?|suvs?)\b"), "auto"),
    (re.compile(r"\buniversit(?:y|ies)\b"), "college"),
    (re.compile(r"\bdrive[\s-]?(?:thru|through|up)\b"), "driveup"),
    (re.compile(r"\bmortgages?\b|\bhome loans?\b(?! refinance)|\bhome loan refinance\b"), "mortgage"),
    (re.compile(r"\bhelocs?\b"), "home equity line of credit"),
    (re.compile(r"\bhsas?\b"), "health savings account"),
    (re.compile(r"\b(?:staff|team) members?\b|\bstaff\b|\bworkers\b"), "employees"),
    (re.compile(r"\bbecom(?:e|ing) an? (?:[a-z]+ )?member\b|\bsign(?:ing)? up for (?:a )?membership\b"), "join member"),
    (re.compile(r"\b(\d+)\s*-?\s*(?:months?|mos?\b\.?)"), r"\1 month "),
    (re.compile(r"\b(\d+)\s*-?\s*(?:years?|yrs?\b\.?)"), r"\1 year "),
    (re.compile(r"\b(?:closes?|closed|closing|closure|shut(?:s|ting)?(?: down)?)\b"), "close"),
    (re.compile(r"\b(?:aba)\b"), "routing"),
    (re.compile(r"\brecogni(?:[sz]|t)\w*"), "recognize"),
    (re.compile(r"\baward\w*"), "award"),
    (re.compile(r"\beligib\w+|\bqualif\w+"), "eligible"),
    (re.compile(r"\bmemberships?\b"), "member"),
    (re.compile(r"\b(?:locations?|located|locator)\b"), "location"),
    (re.compile(r"\b(?:updat|chang)(?:e|es|ed|ing)\b"), "update"),
    (re.compile(r"\bflexib\w+"), "flexible"),
    (re.compile(r"\bintro\b"), "introductory"),
    (re.compile(r"\bjoin(?:s|ed|ing)?\b"), "join"),
    (re.compile(r"\bapys?\b"), "apy"),
    (re.compile(r"\baprs?\b"), "apr"),
    (re.compile(r"\bhigh[\s-]yield\b"), "highyield"),
    (re.compile(r"\bmoney[\s-]market\b"), "moneymarket"),
    (re.compile(r"\b(?:e-?mail)\b"), "email"),
]
WORD = re.compile(r"[a-z0-9$]+(?:[.'’][a-z0-9]+)*")
STOP = set("""
a an and are as at be been being but by can could did do does doing for from had has have having how i if in into is it
its me my of on or our ours please so than that the their them then there these they this those to us was we were what
when where which who whom whose why will with would you your yours about any am s t re ll ve d get got give gives tell
provide show know want need like just also some much many more most out up over per each every all both
credit union cu www com https http current currently today now right still really
""".split())
RATE_META = set("rate apr apy interest dividend yield percent percentage annual pay paying paid earn earning earned offer charge cost published publish list listed".split())
WEAK = set("""loan account saving card credit fixed month year home tier balance term standard regular type new program
    purchase personal online only less older model interest rate cash qualified borrower achieve variable conventional line
    share""".split())
FILLER = set("""
buy buying bought purchase purchasing looking look take taking open opening apply applying use used using new old go going
year month day week time long term balance amount model make made keep carry thinking plan planning want wanted tell
would should could might must may shall able help see find found let say said ask asked asking know expect expected come
put putting keep keeping hold holding finance financing financed borrow borrowing invest investing deposit depositing
going rate bank banking spend option over under above below more than million thousand around about exactly kind sort type good great best better typical average regular normal
""".split())
BASIC_SAVINGS = re.compile(r"(?i)\b(?:share|primary share|regular|basic|statement) savings\b")
PRODUCT_CLASS = set("loan mortgage certificate saving checking card visa moneymarket ira".split())
SEARCH_NOISE = set("should use used using go going want wanted need needed tell say said ask let come".split())
HEAD_NOUNS = set("loan mortgage certificate card account saving checking arm ira line credit".split())


def normalize(text):
    """Lower-case and rewrite into the shared vocabulary.  Used on pages and questions alike."""
    text = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    for pattern, repl in PHRASES:
        text = pattern.sub(repl, text)
    return text


def stem(token):
    """Strip simple plurals (loans -> loan, branches -> branch); a full stemmer over-merges."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes", "sses")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        return token[:-1]
    return token


def tokens_of(text, keep_stop=False):
    """The single tokenizer used everywhere, so indexing and querying always agree."""
    out = []
    for token in WORD.findall(normalize(text)):
        token = token.strip(".'")
        token = re.sub(r"'(?:s|re|ve|ll|d|m|t)$", "", token)
        if not token or (not keep_stop and token in STOP):
            continue
        token = stem(token)
        if keep_stop or (token not in STOP and (len(token) > 1 or token.isdigit())):
            out.append(token)
    return out


IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
FOOTNOTE = re.compile(r"(?<=%)\s?\d\b(?![\d.,%])|(?<=APR)\s?\d\b(?![\d.,%])|(?<=\))\d$|(?<=APY\))\s?\d\b")


def clean_text(md, keep_links=False):
    """Markdown -> plain member-facing text."""
    text = IMAGE.sub("", md)
    if keep_links:
        text = LINK.sub(lambda m: f"{m.group(1)} ({tidy_url(m.group(2))})" if m.group(2).startswith("http") else m.group(1), text)
    else:
        text = LINK.sub(lambda m: m.group(1), text)
    text = text.replace("\\r\\n", " ").replace("\\*", "*").replace("\\|", "|").replace("\\#", "#").replace("\\$", "$")
    text = re.sub(r"[*_`]{1,3}", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = FOOTNOTE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tidy_url(url):
    """Drop tracking parameters and widget fragments so a URL is safe to read out."""
    url = url.split("?")[0]
    if re.search(r"#.*\d{4,}", url):
        url = url.split("#")[0]
    return url


def url_key(url):
    """Host and path, lower-cased, for comparing URLs."""
    parsed = urlparse(url if "//" in url else "https://" + url)
    host = parsed.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return host + parsed.path.rstrip("/").lower()


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
MONTH_RE = "|".join(MONTHS) + "|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
PUBLISHED = re.compile(r"(?im)^\s*(%s)\.? (\d{1,2}), (20\d{2})\s*$" % MONTH_RE)
PUBLISHED_INLINE = re.compile(r"(?i)\\?\| (%s)\.? (\d{1,2}), (20\d{2})" % MONTH_RE)

OFFICIAL, PRODUCT, LANDING, ARCHIVE = 0, 1, 2, 3


def _month_number(name):
    name = name.lower()[:3]
    return next(i + 1 for i, m in enumerate(MONTHS) if m.startswith(name))


# ==============================================================================================
# Page trust
# ==============================================================================================

NEWS_SECTIONS = {"news", "newsroom", "blog", "blogs", "learn", "press", "press-releases", "articles", "insights",
                 "resources", "stories", "media"}
LISTING_SEGMENTS = {"tag", "tags", "author", "authors", "category", "categories", "topics", "page"}
FILE_FOLDERS = {"hubfs", "wp-content", "uploads", "files", "documents", "assets"}


class PageInfo:
    """One page of one snapshot, classified by how far it can be trusted.

    The classification uses only URL conventions that most websites share: news and blog sections,
    calculators, uploaded files, rate pages.  Whether an ordinary page is a "product" page or a
    one-off campaign ("landing") page is decided later by `Site`, from how the site's sections are laid out.
    """
    __slots__ = ("url", "title", "markdown", "tier", "kind", "published", "path")

    def __init__(self, url, title, markdown):
        self.url, self.title, self.markdown = url, title or "", markdown or ""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/").lower()
        segments = [seg for seg in path.split("/") if seg]
        self.path = path
        self.published = None
        if NEWS_SECTIONS & set(segments):
            listing = bool(LISTING_SEGMENTS & set(segments)) or segments[-1] in NEWS_SECTIONS
            self.tier, self.kind = ARCHIVE, ("listing" if listing else "news")
        elif "calculator" in path:
            self.tier, self.kind = ARCHIVE, "calculator"
        elif FILE_FOLDERS & set(segments[:1]) or path.endswith((".pdf", ".xls", ".xlsx", ".doc", ".docx")):
            self.tier, self.kind = ARCHIVE, "file"
        elif parsed.query:
            self.tier, self.kind = ARCHIVE, "duplicate"
        elif any(seg == "rates" or seg.endswith("-rates") for seg in segments):
            self.tier, self.kind = OFFICIAL, "rates"
        else:
            self.tier, self.kind = PRODUCT, "product"
        if self.kind == "news":
            head = self.markdown[:3000]
            match = PUBLISHED.search(head[:1500]) or PUBLISHED_INLINE.search(head)
            if match:
                try:
                    self.published = date(int(match.group(3)), _month_number(match.group(1)), int(match.group(2)))
                except ValueError:
                    self.published = None


# ==============================================================================================
# Rate tables
#
# Rate tables are parsed into structured rows so that a question's qualifiers (term, model year,
# balance, loan amount) can be applied as filters rather than guessed from text similarity.
# ==============================================================================================

TABLE_SEP = re.compile(r"\|[\s|:-]+\|?")
EFFECTIVE = re.compile(r"(?i)rates?\s+(?:are\s+)?effective(?:\s+as\s+of)?\s+(\d{1,2}/\d{1,2}/\d{4})")
DETAILS = re.compile(r"\[(?:See details|Explore|Learn more|Get more info|Compare my options)\]\((https?://[^)\s]+)\)", re.I)
VALUE_HEADER = re.compile(r"(?i)rate|apr|apy|percentage|yield|points|payment|min\.")
TERM = re.compile(r"(\d+)\s*-?\s*(month|year)")
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|million|m\b)?", re.I)
PROSE_RATE = re.compile(r"^(.{3,90}?)\s*:\s*(\d+(?:\.\d+)?\s?%)\s*$")
PCT = re.compile(r"\d+(?:\.\d+)?\s?%")


def _heading_like(line):
    """Return the text if a line acts as a heading above a rate table, else None."""
    if line.startswith("#"):
        return line.lstrip("# ").strip()
    match = re.fullmatch(r"-\s*\[([^\]]+)\]\([^)]*#\)", line)
    if match:
        return match.group(1)
    if len(line) > 70 or line[0] in "[!|*_-✕>" or line.endswith((".", ":", ",")):
        return None
    if re.match(r"\d", line) and (len(line) > 40 or re.match(r"\d[A-Z]", line)):
        return None
    if "http" in line or "%" in line or len(line.split()) > 9 or "effective" in line.lower():
        return None
    return line


class RateRow:
    """One row of a rate table, with everything needed to match and to quote it."""

    __slots__ = ("url", "tier", "group", "product", "header", "cells", "effective", "details", "table",
                 "index", "labels", "values", "minimum", "context", "tokens", "business", "terms",
                 "label_text", "snapshot_id")

    def __init__(self, page, snapshot_id, group, product, header, cells, table, index):
        self.url, self.tier, self.snapshot_id = page.url, page.tier, snapshot_id
        self.group, self.product, self.header, self.cells = group, product, header, cells
        self.table, self.index, self.effective, self.details = table, index, None, None
        self.labels, self.values, self.minimum = [], [], None
        for position, cell in enumerate(cells):
            head = header[position] if position < len(header) else ""
            cell = clean_text(cell)
            is_value = position > 0 and VALUE_HEADER.search(head)
            if not is_value:
                if cell and cell not in ("—", "-", "N/A"):
                    self.labels.append(cell)
                continue
            low = head.lower()
            if "min. to earn" in low:
                self.minimum = cell if re.search(r"\d", cell) else None
            if "points" in low or "payment" in low or "min." in low:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", cell):
                cell += "%"
            if PCT.search(cell):
                self.values.append((_metric(head, cell), cell))
        self.label_text = ", ".join(self.labels)
        self.context = " ".join(x for x in (group, product, self.label_text) if x)
        self.tokens = set(tokens_of(self.context))
        if "mortgage" in self.tokens:
            self.tokens |= {"loan", "home"}
        if self.tokens & {"ira", "roth", "traditional"} and "month" in normalize(self.label_text):
            self.tokens.add("certificate")
        self.business = "business" in self.context.lower() or "/business" in page.path
        self.terms = {(int(n), unit) for n, unit in TERM.findall(normalize(self.label_text))}

    def value_of(self, metric):
        return next((v for m, v in self.values if m == metric), None)

    def headline(self):
        """The number a member means by "the rate": APY for deposits, rate/APR for loans."""
        for metric in ("APY", "rate", "APR", "dividend rate"):
            value = self.value_of(metric)
            if value:
                return value
        return self.values[0][1] if self.values else None


def _metric(header, cell):
    """What a value column measures: APY, APR, dividend rate or plain rate."""
    low = header.lower()
    if "apy" in low or "yield" in low:
        return "APY"
    if "dividend" in low:
        return "dividend rate"
    if "apr" in low or "annual percentage rate" in low or "apr" in cell.lower():
        return "APR"
    return "rate"


def parse_rate_rows(page, snapshot_id):
    """Every row of every rate table on a page, with the headings above it (group, product)
    and the "effective as of" date and "See details" link below it."""
    lines = page.markdown.splitlines()
    rows, table_no = [], 0
    group, product, previous_kind = "", "", None
    pending = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            parsed = [[c.strip() for c in l.strip("|").split("|")] for l in block if not TABLE_SEP.fullmatch(l)]
            pending = []
            if len(parsed) >= 2:
                table_no += 1
                header = [clean_text(c) for c in parsed[0]]
                for index, cells in enumerate(parsed[1:]):
                    row = RateRow(page, snapshot_id, group, product, header, cells, table_no, index)
                    if row.values:
                        rows.append(row)
                        pending.append(row)
            previous_kind = None
            continue
        i += 1
        if not line:
            continue
        heading = _heading_like(line)
        if heading:
            heading = clean_text(heading)
            if line.startswith("- ["):
                group, product, previous_kind = heading, "", "group"
            else:
                if previous_kind == "plain":
                    group = product
                product, previous_kind = heading, "plain"
            pending = []
            continue
        previous_kind = None
        prose = PROSE_RATE.match(clean_text(line)) if page.tier == OFFICIAL else None
        if prose:
            table_no += 1
            row = RateRow(page, snapshot_id, group, product, ["", "APY" if "apy" in prose.group(1).lower() else "APR" if "apr" in prose.group(1).lower() else "rate"],
                          [prose.group(1), prose.group(2)], table_no, 0)
            if row.values:
                rows.append(row)
            continue
        if pending:
            effective, details = EFFECTIVE.search(line), DETAILS.search(line)
            for row in pending:
                if effective and not row.effective:
                    row.effective = effective.group(1)
                if details and not row.details:
                    row.details = details.group(1)
    return rows


def _model_years(label):
    """'2024-27 Model Year' -> (2024, 2027); '2016 Model Year (and older)' -> (0, 2016)."""
    low = label.lower().replace("–", "-")
    if "model year" not in low:
        return None
    match = re.search(r"(20\d{2})\s*-\s*(\d{2,4})", low)
    if match:
        first, second = int(match.group(1)), int(match.group(2))
        second = second if second > 99 else (first // 100) * 100 + second
        return (min(first, second), max(first, second))
    match = re.search(r"(20\d{2})", low)
    if match:
        year = int(match.group(1))
        return (0, year) if "older" in low else (year, year)
    return None


def _money(text):
    """Dollar amounts in text, with k / thousand / million scaled."""
    out = []
    for number, scale in MONEY.findall(text):
        value = float(number.replace(",", ""))
        scale = scale.lower()
        value *= 1000 if scale in ("k", "thousand") else 1_000_000 if scale in ("m", "million") else 1
        out.append(value)
    return out


def _balance_range(label):
    """'$10,000 – $24,999.99' -> (10000, 24999.99); '$25,000-39,999' likewise; '$25,000+' -> (25000, inf)."""
    if "$" not in label or "or less" in label.lower():
        return None
    amounts = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", label.split("$", 1)[1])]
    amounts = [a for a in amounts if a >= 0][:2]
    if not amounts:
        return None
    if len(amounts) >= 2 and amounts[1] > amounts[0]:
        return (amounts[0], amounts[1])
    return (amounts[0], float("inf")) if "+" in label else None


def _loan_cap(label):
    """'$300,000 or less' -> 300000 (loan-amount tiers)."""
    match = re.search(r"\$\s?([\d,]+)\s+or less", label)
    return float(match.group(1).replace(",", "")) if match else None


# ==============================================================================================
# Passages, the per-snapshot index, and search
# ==============================================================================================

DATE_IN_TEXT = re.compile(r"(?i)\b(?:%s)\.? \d{1,2}(?:st|nd|rd|th)?,? (?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b" % MONTH_RE)
TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)
DAY = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?s?"
HOURS_LABEL = re.compile(r"(?im)^[#*_\s]*(lobby|drive[\s-]?(?:up|thru|through))(?:\s+hours)?[*_:\s]*(.*)$")
SCHEDULE_LINE = re.compile(r"(?i)^(%s(?:\s*[-–—]\s*%s)?)\s*:\s*(.+)$" % (DAY, DAY))
ADDRESS = re.compile(r"^(\d+[^,\n]{3,60}),\s*([A-Za-z .']+),\s*([A-Z]{2}),?\s*(\d{5})\b")
US_STATES = ("alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho "
             "illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota "
             "mississippi missouri montana nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah "
             "vermont virginia washington wisconsin wyoming").split()


DEFINITION = re.compile(r"((?:[A-Za-z][\w'’-]*\s+){1,6})\((?:or\s+)?([A-Z]{2,6})\)")


class Passage:
    """A searchable chunk of a page: a paragraph, a bullet list, or one table row with its header."""
    __slots__ = ("page", "text", "section", "terms", "body", "section_terms", "position", "is_row", "section_weight", "repeats")

    def __init__(self, page, text, section, position, is_row):
        self.page, self.text, self.section, self.position, self.is_row = page, text, section, position, is_row
        self.section_terms = frozenset(tokens_of(section))
        words = tokens_of(IMAGE.sub("", text))
        self.body = frozenset(words)
        self.repeats = {w: n for w, n in Counter(words).items() if n > 1} if len(words) != len(self.body) else None
        self.section_weight = 0.0 if _similar(section, page.title.split("|")[0]) else SECTION_WEIGHT
        self.terms = self.body | self.section_terms


class Site:
    """Everything we know about the website as of one snapshot."""

    def __init__(self, snapshot_id, taken_on, pages):
        self.snapshot_id, self.taken_on = snapshot_id, taken_on
        self.origin = {}
        self.legacy = None
        self.pages = sorted(pages, key=lambda p: p.url)
        self.by_url = {p.url: p for p in self.pages}
        self._learn_identity()
        self._mark_landing_pages()
        self._learn_vocabulary()
        self.locator = self._find_locator()
        self.title_terms = {p.url: set(tokens_of(f"{p.title} {urlparse(p.url).path.replace('-', ' ').replace('/', ' ')}")) for p in self.pages}
        self.rate_rows = []
        self.passages = []
        self.postings = defaultdict(list)
        for page in self.pages:
            self.rate_rows.extend(parse_rate_rows(page, self.snapshot_id))
            for position, (section, text, is_row) in enumerate(self._split(page.markdown)):
                passage = Passage(page, text, section, position, is_row)
                if passage.terms:
                    index = len(self.passages)
                    self.passages.append(passage)
                    for term in passage.terms:
                        self.postings[term].append(index)
        self.total = max(1, len(self.passages))
        self.raw_words = set()
        for page in self.pages:
            self.raw_words.update(re.findall(r"[a-z0-9$]+", f"{page.title} {page.markdown}".lower()))
        self.average_length = sum(len(p.body) for p in self.passages) / self.total or 1.0
        self.superseded = set()
        editions = defaultdict(list)
        for page in self.pages:
            if page.kind == "news" and page.published:
                editions[" ".join(tokens_of(page.title.split("|")[0]))].append(page)
        self.editions = [sorted(g, key=lambda p: (p.published, p.url)) for g in editions.values() if len(g) > 1]
        for group in self.editions:
            self.superseded.update(p.url for p in group[:-1])
        self._facts = None
        newest = max((p.published for p in self.pages if p.published), default=None)
        oldest = min((p.published for p in self.pages if p.published), default=None)
        self.news_span = (oldest, newest)
        # The rate sheet: the official pages, plus product-page rows for products the official pages
        # do not list (some sites only show business cards on the card page).  A product-page row that
        # repeats an official row - same words, or same figures for the same named product - is skipped.
        official = [r for r in self.rate_rows if r.tier == OFFICIAL]
        self.sheet = official + [r for r in self.rate_rows if r.tier == PRODUCT and
                                 not any(_repeats(r, o) for o in official)]
        self.sheet_vocab = set().union(*(r.tokens for r in self.sheet)) if self.sheet else set()
        frequency = Counter(t for r in self.sheet for t in r.tokens)
        self.sheet_idf = {t: log((len(self.sheet) + 1) / (n + 1)) + 0.3 for t, n in frequency.items()}

    def _learn_identity(self):
        """The organisation's name, from the site's own host name ("www.dupaco.com" -> "Dupaco")."""
        hosts = Counter(urlparse(p.url).netloc.lower() for p in self.pages)
        self.host = hosts.most_common(1)[0][0] if hosts else ""
        labels = [part for part in self.host.split(".") if part not in ("www", "")]
        self.org = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "")
        spellings = Counter()
        if self.org:
            for page in self.pages[:400]:
                spellings.update(re.findall(r"\b%s\b" % re.escape(self.org), page.title, re.I))
        self.org_name = spellings.most_common(1)[0][0] if spellings else self.org.capitalize()

    def _mark_landing_pages(self):
        """Ordinary pages inside a section of the site (/personal/..., /about/...) are product pages.
        A page that sits alone at the top level and is not in the site's main navigation is a one-off
        campaign page, which is where old promotions tend to linger."""
        sections = Counter(p.path.split("/")[1] for p in self.pages if p.path.count("/") >= 2)
        for page in self.pages:
            if page.tier == PRODUCT and page.path and page.path.count("/") == 1 and sections[page.path[1:]] < 2:
                if not self._in_navigation(page):
                    page.tier, page.kind = LANDING, "landing"

    def _in_navigation(self, page):
        home = next((p for p in self.pages if p.path == ""), None)
        if home is None:
            return False
        return re.search(r"\]\(https?://[^)\s]*%s/?\)" % re.escape(page.path), home.markdown) is not None

    def _learn_vocabulary(self):
        """Learn from the site's own text so questions can use it:
        abbreviations the site defines once ("Volunteer Time Off (VTO)") but rarely uses."""
        self.abbreviations = {}
        for page in self.pages:
            for long_form, short in DEFINITION.findall(page.markdown):
                words = [w for w in re.split(r"[\s-]+", long_form) if w and w.lower() not in ("of", "and", "the")]
                words = words[-len(short):]
                if len(words) == len(short) and "".join(w[0] for w in words).lower() == short.lower():
                    self.abbreviations.setdefault(short.lower(), " ".join(words).lower())

    def _find_locator(self):
        """The branch locator is the trusted page that lists the most street addresses."""
        counts = []
        for page in self.pages:
            if page.tier <= LANDING:
                n = sum(1 for line in page.markdown.splitlines() if ADDRESS.match(line.strip()))
                if n >= 3:
                    counts.append((n, -len(page.path), page.url, page))
        return max(counts)[3] if counts else None

    def strip_org(self, question):
        """The organisation's own name says nothing about which page answers the question."""
        if not self.org:
            return question
        return re.sub(r"(?i)\b%s(?:['’]s)?\b" % re.escape(self.org), " ", question)

    def expand_question(self, q):
        """Spell out abbreviations the site defines but mostly writes in full."""
        for short, long_form in self.abbreviations.items():
            if re.search(r"\b%s\b" % short, q) and self._prefers_long_form(short, long_form):
                q += " " + long_form
        return q

    def _prefers_long_form(self, short, long_form):
        """Expand an abbreviation only when the site mostly writes it out in full."""
        words = tokens_of(long_form)
        if not words or any(w not in self.postings for w in words):
            return False
        with_all = set(self.postings[words[0]]).intersection(*(self.postings[w] for w in words[1:]))
        return len(self.postings.get(short, ())) < len(with_all)

    @staticmethod
    def _split(markdown):
        """Yield (section heading, text, is_table_row). List items under one heading stay together."""
        section, bullets = "", []

        def flush():
            if bullets:
                text = "\n".join(bullets)
                bullets.clear()
                return [(section, text, False)]
            return []

        for block in markdown.split("\n\n"):
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not lines:
                continue
            headings = [l.lstrip("# ").strip() for l in lines if l.startswith("#")]
            table = [l for l in lines if l.startswith("|")]
            if len(table) >= 2:
                yield from flush()
                header = next((l for l in table if not TABLE_SEP.fullmatch(l)), "")
                for row in table:
                    if row != header and not TABLE_SEP.fullmatch(row):
                        yield (section, f"{header}\n{row}", True)
                continue
            if all(l.startswith(("- ", "* ")) for l in lines) and not headings:
                if sum(len(b) for b in bullets) + sum(len(l) for l in lines) > 900:
                    yield from flush()
                bullets.extend(lines)
                continue
            yield from flush()
            if headings:
                section = clean_text(headings[-1])
                body = [l for l in lines if not l.startswith("#")]
                if body:
                    yield (section, "\n".join(body), False)
                else:
                    yield (section, "", False)
            else:
                yield (section, "\n".join(lines), False)
        yield from flush()

    def repeated_facts(self):
        """Figures the organisation restates in many dated articles, with only the number changing.

        Typical source: the "About us" paragraph at the foot of every press release ("It has over 600
        employees and 22 branch office locations"), or a release re-issued under the same headline
        ("the fifth time Forbes recognized ...").  Old copies keep old figures, so these are both
        facts to answer from (newest wins) and contradictions to report.

        Returns {key: [Fact, ...]} where key is the noun the figure describes ("branch", "member").
        """
        if self._facts is not None:
            return self._facts
        sentences = defaultdict(list)          # masked sentence -> [(page, sentence)]
        by_page = defaultdict(list)
        for passage in self.passages:
            page = passage.page
            if page.kind != "news" or not page.published or passage.is_row:
                continue
            if passage.section and passage.section_weight:
                continue                        # a teaser for another article, not this article's text
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", clean_text(passage.text)):
                sentence = sentence.strip(" -")
                if 30 <= len(sentence) <= 400 and FACT_NUMBER.search(sentence):
                    sentences[_mask(sentence)].append((page, sentence))
                    by_page[page.url].append(sentence)
        dated = sum(1 for p in self.pages if p.kind == "news" and p.published)
        minimum = max(REPEATED_MIN_PAGES, int(REPEATED_SHARE * dated))
        occurrences = []
        for masked, found in sentences.items():
            if len({page.url for page, _ in found}) >= minimum:
                occurrences.extend(found)
        for group in self.editions:            # re-issued releases: line up near-identical sentences
            newest = by_page.get(group[-1].url, [])
            for older in group[:-1]:
                for sentence in by_page.get(older.url, []):
                    if self.org not in sentence.lower() or YEAR.search(sentence):
                        continue                # only undated claims about the organisation itself
                    twin = next((n for n in newest if _same_shape(sentence, n)), None)
                    if twin and FACT_NUMBER.findall(twin) != FACT_NUMBER.findall(sentence):
                        occurrences += [(older, sentence), (group[-1], twin)]
        facts = defaultdict(list)
        seen = set()
        for page, sentence in occurrences:
            for fact in _figures(page, sentence):
                if (fact.key, page.url, fact.value) not in seen:
                    seen.add((fact.key, page.url, fact.value))
                    facts[fact.key].append(fact)
        self._facts = {key: sorted(found, key=lambda f: (f.page.published, f.page.url)) for key, found in facts.items()}
        return self._facts

    def idf(self, term):
        return log((self.total + 1) / (len(self.postings.get(term, ())) + 1)) + 1

    def recency(self, page):
        """0..1 for dated pages (1 = newest press release in the snapshot)."""
        oldest, newest = self.news_span
        if not page.published or not newest or newest == oldest:
            return None
        return (page.published - oldest).days / (newest - oldest).days

    def search(self, query_terms, allow=None, required=None, phrase=None, adjust=None):
        """Rank passages. Returns [(score, passage)] best first; ties broken deterministically.

        required: words (names) that should appear in the passage or its page title.
        phrase:   the question's words in order; adjacent pairs found together earn a bonus.
        """
        weights = {t: self.idf(t) for t in query_terms if t in self.postings}
        scores = defaultdict(float)
        for term, weight in weights.items():
            for index in self.postings[term]:
                scores[index] += weight
        ranked = []
        for index, score in scores.items():
            passage = self.passages[index]
            page = passage.page
            if allow and not allow(passage):
                continue
            title_terms = self.title_terms[page.url]
            score = 0.0
            for t, w in weights.items():
                in_body, in_section, in_title = t in passage.body, t in passage.section_terms, t in title_terms
                if in_section and not passage.section_weight:
                    in_section, in_title = False, True
                if in_body and passage.repeats and t in passage.repeats:
                    score += w * REPEAT_WEIGHT * log(passage.repeats[t])
                score += w * (1.0 if in_body else SECTION_WEIGHT if in_section else 0.0)
                score += w * (TITLE_TOO_WEIGHT if in_title and (in_body or in_section) else TITLE_ONLY_WEIGHT if in_title else 0.0)
            score /= LENGTH_BASE + LENGTH_DAMPING * (max(len(passage.body), 1) / self.average_length) ** 0.5
            if required and not required <= (passage.terms | title_terms):
                score *= MISSING_NAME_FACTOR
            score *= TIER_WEIGHT[page.tier]
            if page.kind in ("listing", "duplicate") or page.url in self.superseded:
                score *= LISTING_FACTOR
            elif page.kind in ("calculator", "file"):
                score *= FILE_FACTOR
            recency = self.recency(page)
            if recency is not None:
                score *= RECENCY_FLOOR + RECENCY_SPAN * recency
            if len(passage.text) < 25:
                score *= SHORT_PASSAGE_FACTOR
            if adjust:
                score *= adjust(passage)
            ranked.append((score, passage))
        ranked.sort(key=lambda item: (-item[0], item[1].page.url, item[1].position))
        covered = defaultdict(set)
        for score, passage in ranked[:300]:
            covered[passage.page.url] |= passage.terms & weights.keys()
        head = []
        for score, passage in ranked[:300]:
            elsewhere = covered[passage.page.url] - passage.terms
            head.append((score + PAGE_COVERAGE_BONUS * sum(weights[t] for t in elsewhere), passage))
        head.sort(key=lambda item: (-item[0], item[1].page.url, item[1].position))
        ranked = head + ranked[300:]
        if phrase and len(phrase) >= 2:
            pairs = [(a, b) for a, b in zip(phrase, phrase[1:]) if a in weights and b in weights]
            if pairs:
                head = []
                for score, passage in ranked[:250]:
                    words = tokens_of(f"{passage.section} {passage.text}")
                    adjacent = set(zip(words, words[1:]))
                    bonus = sum(PHRASE_BONUS * (weights[a] + weights[b]) for a, b in pairs if (a, b) in adjacent)
                    head.append((score + bonus * TIER_WEIGHT[passage.page.tier], passage))
                head.sort(key=lambda item: (-item[0], item[1].page.url, item[1].position))
                ranked = head + ranked[250:]
        return ranked


# ==============================================================================================
# Answer wording and question intents
#
# %ORG% is filled in with the organisation's name (learned from the site) when an answer is returned.
# ==============================================================================================

ORG = "%ORG%"
NOT_LISTED = ("I don't know - the website doesn't list {what}. "
              "Please contact %ORG% or visit a branch for a current quote.")
DONT_KNOW = "I don't know - I couldn't find that on the website. Please contact %ORG% for help."

RATE_STRONG = re.compile(r"\b(rates?|apr|apy|interest|dividends?|yield)\b|\d\s?%")
RATE_WEAK = re.compile(r"\b(pay|pays|paying|paid|earn|earns|earning|earned|cost|costs|charge|charges|charging)\b|\bhow much (?:is|are|would|will|does|do|for)\b")
NOT_RATE = re.compile(r"\b(apple pay|google pay|samsung pay|pay off|pay my|pay (?:a |the )?bills?|payments?|pay (?:it|them) (?:back|off)|interest[- ]?free|rate lock|fees?)\b")
LINK_INTENT = re.compile(r"\b(link|url|website|web ?page|web ?site|webpage)\b|\bwhere\b.*\bonline\b|\bonline\b.*\bwhere\b"
                         r"|\bon\s+(?:(?:your|the|its)\s+)?(?:web)?site\b|\bwhere (?:can|do|could|would|should) (?:i|we) (?:see|find|view|check|look)\b")
ONLINE_ACTION = re.compile(r"\b(join|sign up|signup|enroll|apply|open an? (?:account|member))\b.*\bonline\b|\bonline\b.*\b(join|sign up|enroll|apply)\b")
HOURS_INTENT = re.compile(r"\bhours?\b|\bwhat time\b|\bwhen (?:is|are|does|do)\b.*\b(open|lobby|lobbies|driveup)\b|\b(lobby|lobbies|driveup)\b.*\bopen\b"
                          r"|\bopen\b.*\b(?:%s|weekends?|today|tomorrow|late|early)\b|\b(?:are|is)\s+(?:you\s+|the branch\w*\s+|it\s+)?open\b" % DAY)
COUNT_INTENT = re.compile(r"\b(?:how many|number of|count of)\b")
ADDRESS_INTENT = re.compile(r"\b(address|located|location of|where is|where's|where are|directions)\b")
COUNT_WORDS = set("""how many much number count total big large size have has operate run serve served there
employ work worth hold manage""".split())
HOW_TO = re.compile(r"^(?:how (?:can|do|should|would) (?:i|we)|can i|where (?:can|do) i|what do i need to do)\b")
ELIGIBILITY = re.compile(r"\b(eligible|who can|qualif\w+|who is able|requirements?)\b.*\b(join|member)\b|\b(join|member)\b.*\b(eligible|who can|qualif\w+|requirements?)\b")
HOUR_GENERIC = set("""hour lobby lobbies driveup branch branche open close time same general keep typical typically usual
normal standard regular weekday weekend saturday sunday monday tuesday wednesday thursday friday office location
different vary window teller service business operating operation schedule day week morning evening late early""".split())
PLACE_GENERIC = set("""address located location branch branche office where direction find street physical mailing main
headquarter near nearest closest atm""".split())


# ==============================================================================================
# The knowledge system: the three methods the harness calls, and the question router
# ==============================================================================================

class KnowledgeSystem:
    """Keeps one `Site` per snapshot.  Questions are routed to the first handler that recognises
    them; rates come only from the rate sheet; everything else falls back to ranked search."""

    def __init__(self):
        self.sites = []
        self._conflicts = None

    def ingest(self, snapshot: Snapshot) -> None:
        pages = [PageInfo(p.url, p.title, p.markdown) for p in snapshot.pages]
        self.sites.append(Site(snapshot.snapshot_id, snapshot.taken_on, pages))
        self.sites.sort(key=lambda s: s.taken_on)
        # A page missing from a later crawl was not necessarily deleted.  Keep its last known copy
        # in a small "legacy" site, used only when the current snapshot has no confident answer.
        carried = {}
        for site in self.sites:
            present = {p.url for p in site.pages}
            carried = {url: item for url, item in carried.items() if url not in present}
            site.legacy = Site(_carried_id(carried, site), site.taken_on, [page for page, _ in carried.values()]) if carried else None
            if site.legacy:
                site.legacy.origin = {url: sid for url, (_, sid) in carried.items()}
            for page in site.pages:
                carried[page.url] = (page, site.snapshot_id)
        self._conflicts = None

    def answer(self, question, as_of=None):
        # Never let an exception reach the harness: an honest "I don't know" scores better than a crash.
        try:
            result = self._answer(question or "", as_of)
        except Exception:
            result = Answer(text=DONT_KNOW, confident=False)
        result.text = result.text.replace(ORG, self.sites[-1].org_name if self.sites else "us")
        return result

    def conflicts(self):
        if self._conflicts is None:
            try:
                self._conflicts = self._find_conflicts()
            except Exception:
                self._conflicts = []
        return list(self._conflicts)

    def _site_for(self, as_of):
        """The newest snapshot taken on or before `as_of` (the latest one when no date is given)."""
        if not self.sites:
            return None
        if not as_of:
            return self.sites[-1]
        try:
            wanted = date.fromisoformat(str(as_of)[:10])
        except ValueError:
            return self.sites[-1]
        eligible = [s for s in self.sites if date.fromisoformat(s.taken_on) <= wanted]
        return eligible[-1] if eligible else None

    def _answer(self, question, as_of):
        """Route the question.  Narrow, precise handlers go first; general search goes last."""
        site = self._site_for(as_of)
        if site is None:
            return Answer(text="I don't know - I have no copy of the website from that date.", confident=False)
        question = site.strip_org(question)
        q = site.expand_question(normalize(question.strip().rstrip("?.! ")))
        if not tokens_of(q):
            return Answer(text=DONT_KNOW, confident=False)

        # What kind of question is it?  A strong rate question ("APR", "rate", "25-month certificate")
        # is refused if the rate sheet has no row for it; a weak one ("what does it pay?") falls
        # through to search instead.
        link_intent = bool(LINK_INTENT.search(q)) or bool(re.search(r"\b(locator|finder)\b", question.lower()))
        strong_rate = bool(RATE_STRONG.search(q)) and not re.search(r"\binterest(?:ed|ing)\b", q)
        weak_rate = bool(RATE_WEAK.search(q)) and not NOT_RATE.search(q)
        termed_product = bool(TERM.search(q)) and bool(set(tokens_of(q)) & {"certificate", "mortgage", "ira", "special"}) \
            and not re.search(r"\b(penalt\w*|withdraw\w*|calculat\w*|how does|what is a|what's a)\b", q)
        strong_rate = strong_rate or termed_product
        rate_intent = strong_rate or weak_rate

        if "routing" in q:                                   # labelled numbers, voted across pages
            found = self._labelled_number(site, r"routing(?: number)?", r"\d{9}", "routing number")
            if found:
                return found
        if re.search(r"\b(?:phone|telephone)\b.*\bnumber\b|\bnumber to call\b|\bcall (?:you|us)\b|\bcall\s*$", q) and len(tokens_of(q)) <= 5:
            found = self._labelled_number(site, r"\bphone", r"\d{3}-\d{3}-\d{4}", "phone number")
            if found:
                return found
        if link_intent or ONLINE_ACTION.search(q):           # "what's the URL ...", "join online"
            found = self._answer_link(site, q, rate_intent)
            if found:
                return found
        if rate_intent and not link_intent:                  # rates: rate sheet only, or an honest refusal
            found = self._answer_rate(site, question, q, sure=strong_rate)
            if found:
                return found
        if HOURS_INTENT.search(q) and not re.search(r"\bvolunteer\b", q):   # branch hours
            found = self._answer_hours(site, q)
            explicit = bool(re.search(r"\bhours?\b|\bwhat time\b|\blobby|\blobbies|\bdriveup\b", q))
            if found and (found.confident or explicit):
                return found
        found = self._answer_org_fact(site, q)               # "how many branches": newest press release
        if found:
            return found
        if COUNT_INTENT.search(q):                           # other "how many" questions
            found = self._answer_count(site, q)
            if found:
                return found
        if ADDRESS_INTENT.search(q):                         # from the branch locator
            found = self._answer_address(site, q)
            if found:
                return found
        found = self._answer_search(site, q, question)       # everything else
        if not found.confident and site.legacy is not None:  # last resort: pages missing from this crawl
            older = self._answer_search(site.legacy, q, question)
            if older.confident:
                for source in older.sources:
                    source.snapshot_id = site.legacy.origin.get(source.url, source.snapshot_id)
                older.conflict_note = "This page was not in the latest crawl; the answer comes from its last known copy."
                return older
        return found

    def _rate_query(self, site, q):
        """Pull the product words and the qualifiers out of a rate question."""
        words = tokens_of(q)
        terms = {(int(n), unit) for n, unit in TERM.findall(q)}
        years = set()
        if "auto" in words or "model" in words:
            for match in re.finditer(r"\b(19\d{2}|20\d{2})\b", q):
                before = q[max(0, match.start() - 14):match.start()]
                if re.search(r"(?:%s|as of|in|since|during|of)\.?\s*$" % MONTH_RE, before):
                    continue
                years.add(int(match.group(1)))
        amounts = _money(q)
        if not amounts and re.search(r"\bbalance|deposit|borrow|\bloan amount|worth\b", q):
            amounts = [float(n.replace(",", "")) for n in re.findall(r"\b(\d{1,3}(?:,\d{3})+|\d{4,7})\b", q)
                       if float(n.replace(",", "")) not in years and not 1990 <= float(n.replace(",", "")) <= 2035]
        asked_pct = [m.group(0).replace(" ", "") for m in PCT.finditer(q)]
        calendar = set(MONTHS) | {"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mid", "end", "early", "late", "back", "then", "ago"}
        product = [w for w in words if w not in RATE_META and not w.isdigit() and w not in ("month", "year", "model")
                   and w not in calendar and not re.fullmatch(r"[\d.,$%]+k?", w)]
        return {
            "words": words, "product": product, "terms": terms, "years": years, "amounts": amounts,
            "asked_pct": asked_pct,
            "business": bool(re.search(r"\bbusiness|commercial\b", q)),
            "ira": bool(re.search(r"\b(ira|iras|roth|traditional|retirement)\b", q)),
            "refinance": bool(re.search(r"\b(refinance|cash[- ]?out)\b", q)),
            "jumbo": "jumbo" in words,
            "highest": bool(re.search(r"\b(highest|top|maximum|max|best|most)\b", q)),
            "lowest": bool(re.search(r"\b(lowest|minimum|least|cheapest)\b", q)),
            "metric": "APY" if re.search(r"\b(apy|yield)\b", q) else "dividend rate" if "dividend" in q else
                      "APR" if re.search(r"\bapr\b", q) else None,
        }

    def _unlisted_modifier(self, site, q):
        """'construction loan', 'adjustable-rate mortgage', 'FHA loan', '7/1 ARM': a product word the
        rate sheet has never heard of, attached directly to the product noun."""
        vocab = site.sheet_vocab
        sequence = tokens_of(q, keep_stop=True)
        skip = RATE_META | {"month", "year", "home", "equity", "fixed", "mortgage", "auto", "loan", "term"}
        for i, word in enumerate(sequence):
            if word == "arm" and "arm" not in vocab:
                return "ARM"
            if word not in HEAD_NOUNS:
                continue
            j = i - 1
            while j >= 0 and (sequence[j] in skip or sequence[j].isdigit()):
                j -= 1
            if j < 0:
                continue
            candidate = sequence[j]
            if (candidate in STOP or candidate in FILLER or candidate in WEAK or candidate in HEAD_NOUNS
                    or candidate in vocab or len(candidate) < 2 or re.fullmatch(r"[\d.,$%/]+k?", candidate)):
                continue
            return candidate
        return None

    def _match_rows(self, site, query):
        """Rate-sheet rows for the product being asked about, before qualifiers are applied.
        Returns (rows, strong words, reason-if-empty)."""
        wanted = [w for w in dict.fromkeys(query["product"]) if w in site.sheet_vocab]
        strong = [w for w in wanted if w not in WEAK]
        classes = [w for w in wanted if w in PRODUCT_CLASS]
        if not wanted or not (strong or classes):
            return [], strong, "unknown"
        pool = []
        for row in site.sheet:
            low = row.context.lower()
            is_ira = bool(re.search(r"\b(iras?|roth|traditional)\b", low))
            is_refi = bool(re.search(r"refinance|cash-out", low))
            if is_ira != query["ira"] or is_refi != query["refinance"]:
                continue
            if classes and not any(c in row.tokens for c in classes):
                continue
            pool.append(row)

        def score(row):
            return sum(site.sheet_idf.get(w, 0.0) for w in wanted if w in row.tokens)

        scored = [(score(r), r) for r in pool]
        scored = [(s, r) for s, r in scored if s > 0 and (not strong or any(w in r.tokens for w in strong))]
        if not scored:
            return [], strong, "unknown"
        naming_card = bool(set(strong) - {"card", "visa", "mastercard", "credit", "business"})
        card_question = "card" in wanted or "visa" in wanted
        if query["business"]:
            scored = [(s, r) for s, r in scored if r.business]
            if not scored:
                return [], strong, "business"
        else:
            keep = [(s, r) for s, r in scored if not r.business or (card_question and not naming_card and re.search(r"visa|card", r.context.lower()))]
            scored = keep or scored
        best = max(s for s, _ in scored)
        rows = [r for s, r in scored if s >= RATE_MATCH_WINDOW * best]
        if any(r.tier == OFFICIAL for r in rows):
            official_products = {(r.product.lower(), r.business) for r in rows if r.tier == OFFICIAL}
            rows = [r for r in rows if r.tier == OFFICIAL or not any(
                _similar(r.product.lower(), p) for p, b in official_products if b == r.business)]
        uncovered = [w for w in strong if w != "business" and not any(w in r.tokens for r in rows)]
        if uncovered:
            return [], strong, "uncovered"
        return rows, strong, None

    def _rows_by_term(self, site, query):
        """No product named, but the term gives it away: only home loans come in 30 years."""
        wanted = _term_variants(query["terms"])
        rows = []
        for row in site.sheet:
            if row.tier != OFFICIAL:
                continue
            low = row.context.lower()
            if bool(re.search(r"\b(iras?|roth|traditional)\b", low)) != query["ira"] or bool(re.search(r"refinance|cash-out", low)) != query["refinance"]:
                continue
            if row.business != query["business"]:
                continue
            if row.terms & wanted:
                rows.append(row)
        families = {frozenset(row.tokens & PRODUCT_CLASS) for row in rows}
        tables = {(row.url, row.group) for row in rows}
        return rows if rows and len(families) == 1 and len(tables) == 1 else []

    def _refuse_rate(self, site, what, rows=None):
        """The honest answer when the rate sheet has no row for what was asked: no number at all."""
        sources = self._row_sources(rows[:1], site) if rows else self._sheet_sources(site)
        return Answer(text=NOT_LISTED.format(what=what), confident=False, sources=sources)

    def _answer_rate(self, site, question, q, sure=True):
        """Answer a rate question from the rate sheet, or say honestly that it isn't listed.

        sure=False means the wording only hints at a rate ("what does it pay?"): if the rate sheet has
        nothing to say, return None so the question goes on to ordinary search instead of a refusal.
        """
        query = self._rate_query(site, q)
        what = _describe(question)

        # 1. Which product?  Rows of the rate sheet whose words match the question's product words.
        rows, strong, problem = self._match_rows(site, query)
        unlisted = self._unlisted_modifier(site, q)
        if not rows and problem == "unknown" and not unlisted and query["terms"]:
            rows = self._rows_by_term(site, query)
        if not sure:
            content = [w for w in query["product"] if w not in FILLER]
            named = strong or [w for w in content if w in PRODUCT_CLASS]
            if not rows or unlisted or not named or any(w not in site.sheet_vocab for w in content):
                return None

        # 2. Is it something the website does not offer (any more)?
        retired = [w for w in dict.fromkeys(query["product"]) if w not in site.sheet_vocab and w not in WEAK and w not in FILLER
                   and any(w in other.sheet_vocab for other in self.sites if other is not site)]
        if unlisted or not rows or retired:
            fallback = None if (unlisted or retired or query["business"] or problem == "uncovered") else self._rate_from_text(site, query)
            return fallback or self._refuse_rate(site, f"a rate for {what}", rows)

        # 3. Which row(s)?  Each qualifier in the question narrows the rows; if nothing meets a
        #    qualifier, the honest answer is that the site does not list that option.
        if set(strong) <= {"saving", "share"} and not query["business"]:
            rows = [r for r in rows if BASIC_SAVINGS.search(r.context)] or rows
        for narrow in (self._narrow_by_jumbo, self._narrow_by_term, self._narrow_by_model_year,
                       self._narrow_by_amount, self._narrow_by_extreme):
            rows, missing = narrow(rows, query, q, what)
            if missing:
                return self._refuse_rate(site, missing[0], missing[1])

        # 4. Say it.
        if not query["jumbo"]:
            rows = _stable_jumbo_last(rows)
        rows = _dedupe_rows(sorted(rows, key=lambda r: (r.tier, r.business)))[:48]
        text = self._format_rows(rows, query)
        if query["asked_pct"]:      # "Is there a tier that pays 0.35%?" gets a yes or no first
            asked = {_first_pct(p) for p in query["asked_pct"]}
            listed = {_first_pct(v) for r in rows for _, v in r.values}
            top = max((v for v in listed if v is not None), default=None)
            if asked & listed:
                text = "Yes.\n" + text
            elif top is not None:
                text = f"No - the highest rate listed is {top:.2f}%.\n" + text
        return Answer(text=text, sources=self._row_sources(rows, site), conflict_note=self._history_note(rows, site))

    # Each _narrow_by_* step returns (rows, None), or (rows, (what is missing, rows to cite)) to refuse.

    @staticmethod
    def _narrow_by_jumbo(rows, query, q, what):
        if not query["jumbo"]:
            return rows, None
        jumbo = [r for r in rows if "jumbo" in r.context.lower()]
        return (jumbo, None) if jumbo else (rows, (f"a jumbo rate for {what}", None))

    @staticmethod
    def _narrow_by_term(rows, query, q, what):
        with_terms = [r for r in rows if r.terms]
        if not query["terms"] or not with_terms:
            return rows, None
        matched = [r for r in with_terms if r.terms & _term_variants(query["terms"])]
        if matched:
            return matched, None
        n, unit = sorted(query["terms"])[0]
        return rows, (f"a {n}-{unit} option for {what}", with_terms)

    @staticmethod
    def _narrow_by_model_year(rows, query, q, what):
        spans = [(r, _model_years(r.label_text)) for r in rows]
        spans = [(r, span) for r, span in spans if span]
        if not query["years"] or not spans:
            return rows, None
        year = max(query["years"])
        matched = [r for r, span in spans if span[0] <= year <= span[1]]
        newest = max(span[1] for _, span in spans)
        if not matched and year > newest:      # next year's models are priced with the newest tier
            matched = [r for r, span in spans if span[1] == newest]
        return (matched, None) if matched else (rows, (f"a rate for a {year} model year", [spans[0][0]]))

    @staticmethod
    def _narrow_by_amount(rows, query, q, what):
        if not query["amounts"]:
            return rows, None
        amount = query["amounts"][0]
        if re.search(r"\b(under|below|less than|fewer than)\b", q):
            amount -= 0.01
        elif re.search(r"\b(over|above|more than|exceed\w*|at least)\b", q):
            amount += 0.01
        ranged = [(r, _balance_range(r.label_text)) for r in rows]          # deposit balance bands
        if any(span for _, span in ranged):
            rows = [r for r, span in ranged if span and span[0] <= amount <= span[1]] or rows
        capped = [(r, _loan_cap(r.label_text)) for r in rows]               # "$300,000 or less" loan tiers
        if any(cap for _, cap in capped):
            fits = sorted({cap for _, cap in capped if cap and cap >= amount})
            if not fits and all(cap for _, cap in capped):
                return rows, (f"a rate for a loan of that size ({what})", rows)
            if fits:
                rows = [r for r, cap in capped if cap == fits[0] or (cap is None and not query["terms"])]
        return rows, None

    @staticmethod
    def _narrow_by_extreme(rows, query, q, what):
        if not (query["highest"] or query["lowest"]):
            return rows, None
        if not re.search(r"\b(introductory|transfer|promo\w*)\b", q):
            rows = [r for r in rows if "introductory" not in r.tokens] or rows   # an intro rate is not "the lowest APR"
        valued = [(_first_pct(r.headline()), r) for r in rows if r.headline()]
        valued = [(v, r) for v, r in valued if v is not None]
        if not valued:
            return rows, None
        target = max(v for v, _ in valued) if query["highest"] else min(v for v, _ in valued)
        tied = [r for v, r in valued if v == target]
        return (tied[:1] if query["highest"] else tied), None

    def _format_rows(self, rows, query):
        """One block per table (with its effective date), one line per row."""
        tables = defaultdict(list)
        for row in rows:
            tables[(row.url, row.table)].append(row)
        out = []
        for (_, _), members in tables.items():
            first = members[0]
            title = first.product or first.group or "Rates"
            if first.group and first.group.lower() not in title.lower() and title.lower() not in first.group.lower():
                title = f"{first.group} - {title}"
            effective = f" (rates effective as of {first.effective})" if first.effective else ""
            out.append(f"{title}{effective}:")
            for row in members:
                values = row.values
                if query["metric"] == "APY" and row.value_of("APY"):
                    values = [(m, v) for m, v in values if m == "APY"]
                parts = []
                for metric, value in values:
                    parts.append(value if metric.lower() in value.lower() or (metric == "rate" and len(values) == 1) else f"{value} {metric}")
                labels = [l for l in row.labels if l.lower() not in (first.product.lower(), title.lower())] if len(row.labels) > 1 or row.labels and row.labels[0].lower() == first.product.lower() else row.labels
                label = ", ".join(labels)
                line = f"- {label}: " if label else "- "
                line += ", ".join(parts)
                if row.minimum and not _balance_range(row.label_text) and "month" in normalize(row.label_text):
                    line += f" (minimum {row.minimum})"
                out.append(line)
        return "\n".join(out)

    def _row_sources(self, rows, site):
        urls = []
        for row in rows:
            for url in (row.details, row.url):
                if url and url not in urls:
                    urls.append(url)
        return [Source(url=u, snapshot_id=site.snapshot_id) for u in urls[:6]]

    def _sheet_sources(self, site):
        return [Source(url=p.url, snapshot_id=site.snapshot_id) for p in site.pages if p.tier == OFFICIAL][:1]

    def _history_note(self, rows, site):
        """Say when the official value differed in an earlier snapshot (kept out of the answer text)."""
        notes = []
        for row in rows[:6]:
            for older in self.sites:
                if older.taken_on >= site.taken_on:
                    break
                twin = _twin(row, older.sheet)
                if twin and twin.headline() and row.headline() and _first_pct(twin.headline()) != _first_pct(row.headline()):
                    notes.append(f"{row.label_text or row.product}: was {twin.headline()} on {older.taken_on}, now {row.headline()}")
                    break
        return ("Official rate sheet is used over other pages. " + "; ".join(notes)) if notes else None

    def _rate_from_text(self, site, query):
        """A rate that is quoted in prose on a product page (not on the rate sheet)."""
        needed = {w for w in query["product"] if w not in WEAK and w not in FILLER}
        if not needed:
            return None
        ranked = site.search(set(query["product"]), allow=lambda p: p.page.tier <= LANDING and PCT.search(p.text)
                             and needed <= (p.terms | set(tokens_of(p.section))))
        if not ranked:
            return None
        passage = ranked[0][1]
        return Answer(text=_trim(clean_text(passage.text), 500), sources=[Source(url=passage.page.url, snapshot_id=site.snapshot_id)])

    def _answer_link(self, site, q, rate_intent):
        """Questions that want a web address: a rate page, an online action ("join online"), or a page."""
        words = set(tokens_of(q)) - {"link", "url", "website", "web", "page", "site", "online", "webpage", "use", "see", "find", "go"}
        if rate_intent or "rate" in words:
            query = self._rate_query(site, q)
            rows, _, _ = self._match_rows(site, query)
            if rows:
                urls = []
                for row in rows:
                    for url in (row.details, row.url):
                        if url and tidy_url(url).rstrip("/") not in urls:
                            urls.append(tidy_url(url).rstrip("/"))
                official = {url_key(p.url) for p in site.pages if p.tier == OFFICIAL}
                product_page = next((u for u in urls if url_key(u) not in official), None)
                sheet = next((u for u in urls if url_key(u) in official), None)
                lines = []
                if product_page:
                    lines.append(f"You can see those rates at {product_page}#rates")
                if sheet:
                    lines.append(f"All current rates are also listed at {sheet}")
                return Answer(text=". ".join(lines) + ".", sources=[Source(url=u, snapshot_id=site.snapshot_id) for u in urls[:4]])
        if words & {"member", "membership", "join"}:
            words |= {"join", "member"}
        if not words:
            return None
        action = ONLINE_ACTION.search(q)
        verbs = set(tokens_of(action.group(0))) if action else set()
        anchors, found_on = Counter(), defaultdict(list)
        for page in site.pages:
            if page.tier > LANDING:
                continue
            for text, url in LINK.findall(page.markdown):
                if not url.startswith("http") or len(text) > 60:
                    continue
                parsed = urlparse(url)
                anchor_words = set(tokens_of(text)) | set(tokens_of(parsed.netloc.replace(".", " ")))
                hits = words & anchor_words
                if hits and len(hits) >= max(1, len(anchor_words - {site.org}) // 2):
                    link = tidy_url(url).rstrip("/")
                    anchors[link] += len(hits) + 3 * len(hits & verbs)
                    if page.url not in found_on[link]:
                        found_on[link].append(page.url)
        if action and anchors:
            elsewhere = [(n, u) for u, n in anchors.items() if urlparse(u).netloc.lower() != site.host]
            if elsewhere:
                best = max(elsewhere, key=lambda item: (item[0], -len(item[1]), item[1]))[1]
                pages = sorted(found_on[best], key=lambda u: (-len(words & site.title_terms[u]), site.by_url[u].tier, u))
                sources = [Source(url=best, snapshot_id=site.snapshot_id)]
                sources += [Source(url=u, snapshot_id=site.snapshot_id) for u in pages[:1]]
                return Answer(text=f"You can do that online at {best}", sources=sources)
        expanded = set(words)
        if words & {"locator", "atm", "branch", "location", "branche"}:
            expanded |= {"location", "find", "near", "closest", "branch", "atm"}
        ranked = site.search(expanded, allow=lambda p: p.page.tier <= LANDING and p.page.kind != "rates", phrase=tokens_of(q))
        if not ranked:
            return None
        by_page = defaultdict(float)
        for score, passage in ranked[:40]:
            by_page[passage.page.url] += score
        best = max(by_page.items(), key=lambda item: (item[1], item[0]))[0]
        return Answer(text=f"You can find that at {tidy_url(best)}", sources=[Source(url=best, snapshot_id=site.snapshot_id)])

    def _labelled_number(self, site, label, value, name):
        """A labelled number such as "Routing Number: 273974581", voted across pages (trusted pages count more)."""
        votes, where = self._vote_labelled(site, label, value)
        if not votes:
            return None
        best = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        pages = sorted(where[best], key=lambda u: (site.by_url[u].tier, u))[:3]
        note = None
        if len(votes) > 1:
            note = f"Pages disagree on the {name}; {best} appears on the most (and most authoritative) pages."
        return Answer(text=f"%ORG%'s {name} is {best}.", sources=[Source(url=u, snapshot_id=site.snapshot_id) for u in pages], conflict_note=note)

    @staticmethod
    def _vote_labelled(site, label, value):
        pattern = re.compile(r"(?i)%s[^0-9\n]{0,40}?(%s)\b" % (label, value))
        votes, where = Counter(), defaultdict(list)
        for page in site.pages:
            for match in pattern.finditer(page.markdown):
                votes[match.group(1)] += 3 if page.tier <= PRODUCT else 1
                if page.url not in where[match.group(1)]:
                    where[match.group(1)].append(page.url)
        return votes, where

    def _schedules(self, site):
        """[(page, kind, [(days, hours)])] for every lobby / drive-up schedule published."""
        found = []
        for page in site.pages:
            if page.kind == "listing" or not HOURS_LABEL.search(page.markdown):
                continue
            kind, current = None, []
            for raw in page.markdown.splitlines():
                line = raw.strip()
                if not line:
                    continue
                label = HOURS_LABEL.match(line)
                if label:
                    if kind and current:
                        found.append((page, kind, current))
                    kind = "lobby" if label.group(1).lower() == "lobby" else "driveup"
                    current, line = [], label.group(2).strip()
                    if not line:
                        continue
                match = SCHEDULE_LINE.match(line)
                if kind and match and (TIME.search(match.group(2)) or re.search(r"(?i)closed|atm only", match.group(2))):
                    days = re.sub(r"\s*[-–—]\s*", "-", match.group(1))
                    hours = re.sub(r"\s*[-–—]\s*", "-", match.group(2).strip())
                    hours = hours if re.search(r"m\.$", hours) else hours.rstrip(".")
                    current.append((days, hours))
                elif kind and current:
                    found.append((page, kind, current))
                    kind, current = None, []
            if kind and current:
                found.append((page, kind, current))
        return found

    def _answer_hours(self, site, q):
        """Branch hours.  A named branch gets its own schedule (or a refusal); otherwise the most
        common schedule plus the variants, and a question back: hours genuinely differ by branch."""
        words = set(tokens_of(q))
        kind = "driveup" if "driveup" in words else "lobby" if words & {"lobby", "lobbies"} else None
        place = {w for w in words if w not in HOUR_GENERIC and w not in FILLER and not w.isdigit()}
        schedules = self._schedules(site)
        if not schedules:
            return None
        if place:
            matching = [s for s in schedules if place <= (site.title_terms[s[0].url] | set(tokens_of(s[0].markdown[:4000])))]
            matching = [s for s in matching if place & site.title_terms[s[0].url] or
                        all(w in tokens_of(s[0].markdown[:1500]) for w in place)]
            if not matching:
                where = " ".join(w for w in tokens_of(q) if w in place)
                return Answer(text=f"I don't know - I couldn't find published hours for that location ({where}). "
                                   "Please contact %ORG% or check the branch locator.", confident=False,
                              sources=self._contact_source(site))
            chosen = [s for s in matching if kind in (None, s[1])] or matching
            newest = max(chosen, key=lambda s: (s[0].published or date.min, s[0].url))[0]
            chosen = [s for s in chosen if s[0] is newest]
            lines = []
            for _, k, entries in chosen:
                lines.append(("Lobby: " if k == "lobby" else "Drive-up: ") + "; ".join(f"{d} {h}" for d, h in entries))
            return Answer(text=f"{clean_text(newest.title)} lists these hours:\n" + "\n".join(lines),
                          sources=[Source(url=newest.url, snapshot_id=site.snapshot_id)])
        kinds = [kind] if kind else ["lobby", "driveup"]
        lines, pages = [], []
        for k in kinds:
            variants = Counter()
            first_seen = {}
            for page, sk, entries in schedules:
                if sk != k:
                    continue
                text = "; ".join(f"{d} {h}" for d, h in entries)
                variants[text] += 1
                first_seen.setdefault(text, page)
            if not variants:
                continue
            ordered = sorted(variants.items(), key=lambda item: (-item[1], item[0]))
            name = "Lobby" if k == "lobby" else "Drive-up"
            lines.append(f"{name} hours vary by branch. The most common schedule is {ordered[0][0].rstrip('.')}.")
            for text, _ in ordered[1:4]:
                lines.append(f"Other branches: {text.rstrip('.')}.")
            for text, _ in ordered[:4]:
                if first_seen[text].url not in pages:
                    pages.append(first_seen[text].url)
        if not lines:
            return None
        lines.append("Which branch would you like hours for?")
        sources = [Source(url=u, snapshot_id=site.snapshot_id) for u in pages[:5]] + self._contact_source(site)
        home = next((p for p in site.pages if p.path == "" and p.kind == "product"), None)
        if home:
            sources.append(Source(url=home.url.rstrip("/") + "/", snapshot_id=site.snapshot_id))
        return Answer(text="\n".join(lines), sources=sources,
                      conflict_note="Branch schedules differ between press releases; all published variants are listed.")

    def _contact_source(self, site):
        return [Source(url=site.locator.url, snapshot_id=site.snapshot_id)] if site.locator else []

    def _answer_address(self, site, q):
        """A branch address, matched against the branch locator by city, street or name."""
        words = set(tokens_of(q))
        place = {w for w in words if w not in PLACE_GENERIC and w not in FILLER and not w.isdigit()}
        if not place:
            return None
        page = site.locator
        if not page:
            return None
        found, heading, kind = [], "", ""
        lines = [l.strip() for l in page.markdown.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            if line.startswith("###"):
                heading = clean_text(line)
                kind = lines[i + 1] if i + 1 < len(lines) else ""
                continue
            match = ADDRESS.match(line)
            if not match:
                continue
            city_words = set(tokens_of(match.group(2)))
            street_words = set(tokens_of(match.group(1)))
            heading_words = set(tokens_of(heading)) - {"branch"}
            if place <= city_words | street_words | heading_words:
                own_branch = site.org in kind.lower() and "branch" in kind.lower()
                entry = (0 if own_branch else 1, f"{match.group(1)}, {match.group(2)}, {match.group(3)} {match.group(4)}", kind)
                if entry not in found:
                    found.append(entry)
        if not found:
            return None
        branches = [e for e in found if e[0] == 0] or found
        if len(branches) == 1:
            text = f"The {' '.join(w.capitalize() for w in sorted(place))} branch is at {branches[0][1]}."
        else:
            text = "%ORG% locations matching that:\n" + "\n".join(f"- {e[1]} ({clean_text(e[2])})" for e in branches[:8])
        return Answer(text=text, sources=[Source(url=page.url, snapshot_id=site.snapshot_id)])

    def _answer_org_fact(self, site, q):
        """'How many branches / members / employees?' - from the figures the site restates in its
        press releases; the most recent release wins."""
        counting = bool(COUNT_INTENT.search(q)) or bool(re.search(r"\bhow (?:big|large)\b|\btotal\b|\bsize\b", q))
        if not counting:
            return None
        words = set(tokens_of(q)) - COUNT_WORDS
        if not words:
            return None
        facts = site.repeated_facts()
        groups = [found for found in facts.values() if words <= set().union(*(f.label for f in found))]
        if not groups:
            return None
        found = max(groups, key=lambda g: (len(g), g[-1].page.published))
        newest = found[-1]
        older = list(dict.fromkeys(f.value for f in reversed(found) if f.value != newest.value))
        note = None
        if older:
            note = f"Older press releases say {', '.join(older[:6])}; the most recent one ({newest.page.published.isoformat()}) is used."
        text = f"According to %ORG%'s most recent press release ({newest.page.published:%B} {newest.page.published.day}, {newest.page.published.year}): {newest.sentence}"
        return Answer(text=text, sources=[Source(url=newest.page.url, snapshot_id=site.snapshot_id)], conflict_note=note)

    def _answer_count(self, site, q):
        """Other "how many" questions: sentences with a number before the noun.  Figures reported per
        year are given with their year; otherwise the most recent statement wins."""
        match = re.search(r"\b(?:how many|number of|count of)\s+(.*)$", q)
        if not match:
            return None
        tail = tokens_of(match.group(1))
        if not tail:
            return None
        noun = tail[0]
        others = {w for w in tail[1:] if w not in FILLER and w not in ("last", "year", "have", "operate", "run", "location", "office", "total")}
        number = re.compile(r"(?i)\b(\d[\d,]*)\+?\s+(?:[A-Za-z'’-]+\s+){0,2}?%s\w*" % re.escape(noun[:5]))
        hits = []
        for page in site.pages:
            if page.kind in ("listing", "duplicate", "file", "calculator"):
                continue
            text = clean_text(page.markdown)
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                found = number.search(sentence)
                if not found or len(sentence) > 600:
                    continue
                sentence_words = set(tokens_of(sentence))
                for short in sentence_words & site.abbreviations.keys():
                    sentence_words.update(tokens_of(site.abbreviations[short]))
                if others and len(others & sentence_words) < max(1, (len(others) + 1) // 2):
                    continue
                value = found.group(1)
                if value in ("24",) and re.search(r"(?i)24[- ]hour", sentence):
                    continue
                stated = re.search(r"\b(?:in|during)\s+((?:19|20)\d{2})\b", sentence)
                hits.append((page.published, page, sentence.strip(), value, stated.group(1) if stated else None))
        if not hits:
            return None
        dated = [h for h in hits if h[0]]
        yearly = {}
        for published, page, sentence, value, stated in sorted(dated, key=lambda h: (h[0], h[1].url)):
            if stated:
                yearly[stated] = (page, sentence, value)
        if len(yearly) >= 2:
            lines, sources = [], []
            for year in sorted(yearly, reverse=True)[:3]:
                page, sentence, value = yearly[year]
                lines.append(f"In {year}: {_trim(sentence.lstrip('- '), 260)}")
                sources.append(Source(url=page.url, snapshot_id=site.snapshot_id))
            return Answer(text="\n".join(lines), sources=sources)
        pool = dated or hits
        newest = max(pool, key=lambda h: (h[0] or date.min, h[1].tier == PRODUCT, h[1].url))
        published, page, sentence, value, _ = newest
        values = Counter(h[3] for h in pool)
        note = None
        if len(values) > 1:
            note = (f"Older pages give other figures ({', '.join(sorted(values, key=lambda v: (-values[v], v))[:5])}); "
                    f"the most recent statement ({published.isoformat() if published else 'undated'}) is used.")
        phrase = re.search(r"(?i)\b%s\+?(?:\s+[A-Za-z'’-]+){1,4}" % re.escape(value), sentence)
        what = phrase.group(0).strip() if phrase else f"{value} {noun}"
        what = re.sub(r"(?i)\s+(?:and|in|that|which|who|to|for|with|as|at|by|from)$", "", what)
        text = f"%ORG% has {what}, according to the most recent statement on the website."
        return Answer(text=text, sources=[Source(url=page.url, snapshot_id=site.snapshot_id)], conflict_note=note)

    def _answer_search(self, site, q, question=""):
        """Everything without a dedicated handler: rank passages, take the best page, and answer only
        if what we would return actually covers the question."""
        words = [w for w in tokens_of(q) if w not in SEARCH_NOISE] or tokens_of(q)
        query = self._search_terms(words, q)
        if not query:
            return Answer(text=DONT_KNOW, confident=False)
        names = _proper_nouns(question)
        if any(w not in site.raw_words for w in _proper_nouns(question, raw=True)):
            return Answer(text=DONT_KNOW, confident=False)      # asks about something the site never mentions

        ranked = site.search(query, required=names or None, phrase=words, adjust=self._question_adjuster(q))
        if not ranked:
            return Answer(text=DONT_KNOW, confident=False)
        best_score, best = self._prefer_newest_release(ranked)
        page = best.page
        title_terms = site.title_terms[page.url]
        chosen = self._answer_passages(site, ranked, best, best_score)
        refusal = Answer(text=DONT_KNOW, confident=False, sources=[Source(url=page.url, snapshot_id=site.snapshot_id)])

        # Confidence: the passages we would return must cover enough of the question, weighted by
        # how rare each word is, and must mention any name the question asked about.
        eligibility = bool(ELIGIBILITY.search(q))
        if _coverage(site, set(words), set(title_terms).union(*(p.terms for p in chosen))) < (
                MIN_COVERAGE_SHORT if len(set(words)) <= 2 else MIN_COVERAGE) and not eligibility:
            return refusal
        if names and not names <= (best.terms | title_terms):
            named = next((p for _, p in ranked[:400] if p.page is page and names <= p.terms), None)
            if named is None:
                return refusal
            if named not in chosen:
                chosen.insert(1, named)
        if eligibility:                     # add the passage that names the states the charter covers
            states = next((p for p in site.passages if p.page is page and p not in chosen
                           and sum(st in p.text.lower() for st in US_STATES) >= 2), None)
            if states:
                chosen.append(states)

        text = self._compose(page, chosen, keep_links=bool(re.search(r"\b(how|where|online|apply|join|sign|contact|link)\b", q)))
        sources = [Source(url=page.url, snapshot_id=site.snapshot_id)]
        if HOW_TO.search(q):                # account-service questions: always say how to reach a person
            phone = self._contact_phone(site)
            if phone and phone not in text:
                text += f"\nYou can also call %ORG% at {phone} for help."
                sources += self._contact_source(site)
        return Answer(text=text, sources=sources)

    @staticmethod
    def _search_terms(words, q):
        query = set(words)
        if "open" in query and "branch" in query:
            query.add("close")              # "is the branch open?" is answered by its closing notice
        if "wallet" in query:
            query |= {"apple", "google", "samsung", "pay"}
        if ELIGIBILITY.search(q):
            query |= {"join", "eligible"}
        return query

    @staticmethod
    def _question_adjuster(q):
        """Score adjustments that depend on what kind of question this is."""
        wants_date = bool(re.search(r"\bwhen\b|\bwhat (?:date|day)\b|\bdate\b|\bhow long\b|\bstill\b", q))
        how_to = bool(HOW_TO.search(q))
        eligibility = bool(ELIGIBILITY.search(q))

        def adjust(passage):
            factor = 1.0
            page = passage.page
            if page.kind == "news" and passage.section and not _similar(passage.section, page.title.split("|")[0]):
                factor *= TEASER_FACTOR
            if wants_date and DATE_IN_TEXT.search(passage.text):
                factor *= DATED_PASSAGE_FACTOR
            if eligibility and {"eligible", "join"} <= passage.section_terms:
                factor *= ELIGIBILITY_FACTOR
            if how_to and page.tier == ARCHIVE:
                factor *= HOW_TO_ARCHIVE_FACTOR
            return factor
        return adjust

    @staticmethod
    def _prefer_newest_release(ranked):
        """When several press releases answer about equally well, the newest describes the present."""
        best_score, best = ranked[0]
        if best.page.kind != "news":
            return best_score, best
        rivals = {}
        for score, passage in ranked[:200]:
            if score < NEWEST_RELEASE_WINDOW * best_score:
                break
            if passage.page.kind == "news" and passage.page.published and passage.page.url not in rivals:
                rivals[passage.page.url] = (score, passage)
        if len(rivals) > 1:
            return max(rivals.values(), key=lambda item: (item[1].page.published, item[0]))
        return best_score, best

    @staticmethod
    def _answer_passages(site, ranked, best, best_score):
        """The best passage plus the strongest other passages from the same page (for a press
        release: its opening paragraph, and never teasers for other releases)."""
        page = best.page
        same_page = [(sc, p) for sc, p in ranked[:400] if p.page is page and p is not best and sc >= 0.5 * best_score and len(p.text) >= 40]
        if page.kind == "news":
            same_page = [(sc, p) for sc, p in same_page if not p.section_weight]
        chosen = [best] + [p for _, p in same_page[:4 if page.kind == "news" else 3]]
        if page.kind == "news":
            lede = next((p for p in site.passages if p.page is page and len(p.text) >= 120 and not p.is_row
                         and not p.text.startswith(("[", "!", "#")) and _similar(p.section, page.title.split("|")[0])), None)
            if lede and lede not in chosen:
                chosen.insert(1, lede)
        return chosen

    @staticmethod
    def _compose(page, chosen, keep_links):
        """Readable answer text: page title, then each passage labelled with its sub-heading."""
        title = clean_text(page.title).split("|")[0].strip()
        parts, seen, sections = [], set(), set()
        for passage in chosen:
            body = passage.text.split("\n", 1)[1] if passage.is_row and "\n" in passage.text else passage.text
            body = clean_text(body, keep_links)
            if passage.is_row:
                body = " | ".join(c.strip() for c in body.strip("|").split("|"))
            if not body or body.lower() in seen:
                continue
            seen.add(body.lower())
            section = passage.section
            if (section and len(section) < 90 and section.lower() not in sections and section.lower() not in body.lower()
                    and not _similar(section, title)):
                sections.add(section.lower())
                body = f"{section}: {body}"
            parts.append(_trim(body, 480))
        text = "\n".join(parts)
        if title and title.lower() not in text.lower()[:len(title) + 5]:
            text = f"{title}\n{text}"
        return _trim(text, 2200 if page.kind == "news" else 1500)

    def _contact_phone(self, site):
        votes, _ = self._vote_labelled(site, r"\bphone", r"\d{3}-\d{3}-\d{4}")
        return max(votes.items(), key=lambda item: (item[1], item[0]))[0] if votes else None

    def _find_conflicts(self):
        """All contradictions in the latest snapshot: rates, restated press-release figures, branch hours."""
        if not self.sites:
            return []
        latest = self.sites[-1]
        out = []
        out.extend(self._rate_conflicts(latest))
        out.extend(self._statement_conflicts(latest))
        out.extend(self._hours_conflicts(latest))
        return out

    def _rate_conflicts(self, latest):
        """Rates that disagree with today's official rate sheet.

        Two sources of disagreement: (a) other pages of the latest snapshot that still show an old copy
        of a rate table (calculators, campaign pages), and (b) the rate sheet in earlier snapshots.
        Tables are matched by structure (terms, tiers, balance bands) rather than by wording, so a
        renamed heading or a relabelled model-year tier does not hide a conflict.
        """
        current = {k: v for k, v in _tables([r for r in latest.sheet if r.tier == OFFICIAL]).items() if not _derived(v)}
        facts, retired, lingering = self._collect_rate_disagreements(latest, current)

        # One conflict per question a member would ask: all 30-year mortgage tiers together, each
        # newest-vehicle auto term on its own, older model years together.
        merged = {}
        for (key, _), fact in facts.items():
            if not fact["seen"]:
                continue
            entry = merged.setdefault(_conflict_topic(fact["row"], current[key]),
                                      {"rows": [], "seen": {}, "stale": set(), "changed": set()})
            entry["rows"].append(fact["row"])
            for text, sources in fact["seen"].items():
                entry["seen"].setdefault(text, []).extend(sources)
            entry["stale"] |= fact["stale"]
            entry["changed"] |= fact["changed"]
        built = [self._rate_conflict(topic, entry, latest) for topic, entry in merged.items()]
        found = [c for _, c in sorted((item for item in built if item), key=lambda item: item[0])]

        # An offer the rate sheet dropped, but another page still advertises.
        for key, row in lingering.items():
            old_row, last_seen = retired[key]
            found.append(Conflict(
                topic=f"{clean_text(row.label_text)} (offer no longer on the rate sheet)",
                values=[_value_text(row), "no longer offered"],
                sources=[Source(url=row.url, snapshot_id=latest.snapshot_id, quote=f"{row.label_text}: {_value_text(row)}"),
                         Source(url=old_row.url, snapshot_id=old_row.snapshot_id, quote=f"last listed on the rate sheet {last_seen}")],
                resolution="no longer offered",
                reason=f"the official rate sheet dropped this row after {last_seen}; {urlparse(row.url).path} was never updated"))

        # "Up to X% APY" promises above anything the rate sheet pays, unless already covered above.
        known = [{_first_pct(p) for v in c.values for p in PCT.findall(v)} for c in found]
        for extra in self._advertised_conflicts(latest):
            if not any({_first_pct(v) for v in extra.values} <= k for k in known):
                found.append(extra)
        return found

    def _collect_rate_disagreements(self, latest, current):
        """For every row of today's official tables, the different values seen elsewhere.
        Also returns rows the rate sheet used to have (retired) and copies of them still shown (lingering)."""
        facts = {}
        for key, rows in current.items():
            for row in rows:
                facts[(key, _signature(row, rows))] = {"row": row, "seen": {}, "stale": set(), "changed": set()}
        retired, lingering = {}, {}

        def record(table_rows, snapshot_id, taken_on, stale):
            match = _match_table(table_rows, current)
            if match is None:
                return
            for row in table_rows:
                signature = _signature(row, table_rows)
                fact = facts.get((match, signature))
                if fact is None:
                    if not stale:
                        retired[(match, signature)] = (row, taken_on)
                    elif (match, signature) in retired:
                        lingering.setdefault((match, signature), row)
                    continue
                now, then = _metric_values(fact["row"]), _metric_values(row)
                if not then or not now or then == now or (stale and then <= now):
                    continue
                text = _value_text(row)
                fact["seen"].setdefault(text, []).append(
                    Source(url=row.url, snapshot_id=snapshot_id, quote=f"{row.label_text}: {text}"))
                if stale:
                    fact["stale"].add(urlparse(row.url).path)
                else:
                    fact["changed"].add(taken_on)

        for older in self.sites[:-1]:
            for rows in _tables([r for r in older.sheet if r.tier == OFFICIAL]).values():
                if not _derived(rows):
                    record(rows, older.snapshot_id, older.taken_on, False)
        for rows in _tables([r for r in latest.rate_rows if r.tier != OFFICIAL]).values():
            if not _derived(rows):
                record(rows, latest.snapshot_id, latest.taken_on, True)
        return facts, retired, lingering

    @staticmethod
    def _rate_conflict(topic, entry, latest):
        """(sort order, Conflict) for one merged topic, or None if nothing actually differs."""
        rows = entry["rows"]
        now = list(dict.fromkeys(_value_text(r) for r in rows))
        old = [t for t in entry["seen"] if t not in now]
        if not old:
            return None
        reasons = []
        if entry["stale"]:
            reasons.append("the official rate sheet is authoritative; %s still shows an outdated copy of the table"
                           % ", ".join(sorted(entry["stale"])[:3]))
        if entry["changed"]:
            reasons.append(f"the rate sheet itself changed after {max(entry['changed'])}; "
                           f"the newest snapshot ({latest.taken_on}) is current")
        first = rows[0]
        effective = f" (effective {first.effective})" if first.effective else ""
        sources = [Source(url=first.url, snapshot_id=latest.snapshot_id, quote=f"{first.label_text}: {now[0]}{effective}")]
        sources += [entry["seen"][text][0] for text in old]
        return (0 if entry["stale"] else 1, Conflict(
            topic=topic, values=now + old, sources=sources[:8],
            resolution="; ".join(now) if len(now) <= 4 else now[0], reason="; ".join(reasons)))

    def _advertised_conflicts(self, latest):
        """Marketing copy that still promises "up to X% APY" when no row of the rate sheet pays that much."""
        out = []
        pattern = re.compile(r"(?i)up to\s+\**(\d+(?:\.\d+)?)\s?%\s*(?:apy|annual percentage yield)")
        for word, label in (("checking", "checking"), ("certificate", "certificate"), ("saving", "savings")):
            rows = [r for r in latest.sheet if r.tier == OFFICIAL and not r.business and word in r.tokens and r.value_of("APY")]
            if not rows:
                continue
            best = max(rows, key=lambda r: _first_pct(r.value_of("APY")) or 0)
            top = _first_pct(best.value_of("APY"))
            claims = {}
            for page in latest.pages:
                if page.tier == OFFICIAL or page.kind in ("news", "listing", "duplicate", "file"):
                    continue
                for match in pattern.finditer(page.markdown):
                    around = tokens_of(page.markdown[max(0, match.start() - 200):match.end() + 200])
                    if word not in around:
                        continue
                    value = float(match.group(1))
                    if value > top + 1e-9:
                        claims.setdefault(f"{value:.2f}%", Source(url=page.url, snapshot_id=latest.snapshot_id,
                                                                  quote=clean_text(match.group(0))))
            if claims:
                current = best.value_of("APY")
                out.append(Conflict(
                    topic=f"highest {label} APY advertised ({best.product})",
                    values=[current] + sorted(claims),
                    sources=[Source(url=best.url, snapshot_id=latest.snapshot_id, quote=f"{best.label_text}: {current}")]
                    + [claims[k] for k in sorted(claims)],
                    resolution=current,
                    reason="the official rate sheet is authoritative; the promotional page quotes a rate that is no longer offered"))
        return out

    def _statement_conflicts(self, latest):
        """A figure the site restates in many releases with different values: the newest release wins."""
        out = []
        for key, found in sorted(latest.repeated_facts().items()):
            values = list(dict.fromkeys(f.value for f in reversed(found)))
            if len(values) < 2:
                continue
            newest = found[-1]
            latest_by_value = {}
            for fact in reversed(found):
                latest_by_value.setdefault(fact.value, fact)
            shown = [latest_by_value[v] for v in values[:6]]
            out.append(Conflict(
                topic=f"{newest.text.replace(newest.value, '…', 1) if newest.value in newest.text else newest.text} "
                      "(figure repeated across press releases)",
                values=[f.text for f in shown],
                sources=[Source(url=f.page.url, snapshot_id=latest.snapshot_id, quote=f.sentence) for f in shown],
                resolution=newest.text,
                reason=f"stated in the most recent dated press release ({newest.page.published.isoformat()}); "
                       "older releases repeat outdated figures",
            ))
        return out

    def _hours_conflicts(self, latest):
        """Different published schedules.  Reported, but the reason says plainly that hours differ by branch."""
        out = []
        schedules = self._schedules(latest)
        for kind, name in (("lobby", "lobby hours"), ("driveup", "drive-up hours")):
            variants, pages = Counter(), {}
            for page, k, entries in schedules:
                if k == kind:
                    text = "; ".join(f"{d} {h}" for d, h in entries)
                    variants[text] += 1
                    pages.setdefault(text, page)
            if len(variants) < 2:
                continue
            ordered = sorted(variants.items(), key=lambda item: (-item[1], item[0]))
            out.append(Conflict(
                topic=f"branch {name}",
                values=[text for text, _ in ordered],
                sources=[Source(url=pages[text].url, snapshot_id=latest.snapshot_id, quote=text) for text, _ in ordered],
                resolution=ordered[0][0],
                reason="hours differ by branch rather than being wrong; the most common schedule is given and the member is asked which branch",
            ))
        return out


QUESTION_WORDS = set("what when where which who whom whose why how is are was were do does did can could will would should may i tell give show provide get list please has have had".split())


# ==============================================================================================
# Figures restated across press releases (see Site.repeated_facts)
# ==============================================================================================

ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
FACT_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?%?(?:\s+(?:billion|million))?|\b(?:" + "|".join(ORDINALS) + r")\b", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
CLAUSE_BREAK = r",\s|;|\.\s|\.$| and | with | by "


class Fact:
    """One figure in one sentence of one dated article, with the words that say what it measures."""
    __slots__ = ("key", "label", "value", "text", "page", "sentence")

    def __init__(self, key, label, value, text, page, sentence):
        self.key, self.label, self.value, self.text, self.page, self.sentence = key, label, value, text, page, sentence


def _mask(sentence):
    return FACT_NUMBER.sub("#", sentence.lower())


def _same_shape(a, b):
    """Two sentences that say the same thing apart from their figures and small words
    ("It was the fourth time ..." / "This was the fifth time ...")."""
    if len(FACT_NUMBER.findall(a)) != len(FACT_NUMBER.findall(b)):
        return False
    words_a, words_b = set(tokens_of(_mask(a))), set(tokens_of(_mask(b)))
    return bool(words_a) and words_a == words_b


def _figures(page, sentence):
    """Each figure in a sentence, labelled by the content words right after it ("22 branch office
    locations") or, when none follow, right before it ("assets exceeding $3.6 billion")."""
    out = []
    for match in FACT_NUMBER.finditer(sentence):
        value = match.group(0).lower()
        before_text = sentence[max(0, match.start() - 12):match.start()].lower()
        if YEAR.fullmatch(value) or re.search(r"(?:%s)\.?\s*$" % MONTH_RE, before_text):
            continue                            # a date, not a figure
        clause_after = re.split(CLAUSE_BREAK, sentence[match.end():], 1)[0]
        after = tokens_of(clause_after)[:3]
        if after:
            label_words, text = after, f"{match.group(0)}{clause_after.rstrip()}"
        else:
            clause_before = re.split(CLAUSE_BREAK, sentence[:match.start()])[-1]
            label_words = tokens_of(clause_before)[-3:]
            text = f"{clause_before.strip()} {match.group(0)}"
        if label_words:
            out.append(Fact(label_words[0], set(label_words), value, text.strip(), page, sentence))
    return out


def _proper_nouns(question, raw=False):
    """Names in the question ("Forbes", "Hy-Vee", "Mars"): capitalised words that are not just sentence case."""
    names = set()
    words = re.findall(r"[A-Za-z][A-Za-z$'’-]*", question)
    for position, word in enumerate(words):
        if not word[0].isupper() or word.isupper():
            continue
        if position == 0 and (word.lower() in QUESTION_WORDS or word.lower() in STOP or len(words) > 1 and not words[1][0].isupper()):
            continue
        if word.lower().strip("'’s") in ("i", "apr", "apy") or word.lower() in MONTHS:
            continue
        if raw:
            names.update(w for w in re.findall(r"[a-z0-9$]+", word.lower().replace("’", "'").split("'")[0]) if len(w) > 2)
        else:
            names.update(t for t in tokens_of(word) if len(t) > 2)
    return names


# ==============================================================================================
# Small helpers
# ==============================================================================================

def _carried_id(carried, site):
    return max((sid for _, sid in carried.values()), default=site.snapshot_id)


def _repeats(row, official):
    """True if a product-page row only repeats an official row."""
    if row.tokens <= official.tokens:
        return True
    names = (row.tokens & official.tokens) - WEAK - PRODUCT_CLASS - RATE_META
    return bool(names) and _metric_values(row) == _metric_values(official)


def _tables(rows):
    """Group rate rows by the table they came from."""
    tables = {}
    for row in rows:
        tables.setdefault((row.snapshot_id, row.url, row.table), []).append(row)
    return tables


def _derived(rows):
    """Business, IRA and refinance tables mirror a primary table; reporting them again adds no information."""
    context = normalize(" ".join(f"{r.group} {r.product} {r.label_text}" for r in rows[:3]))
    return rows[0].business or bool(re.search(r"\b(iras?|roth|traditional|refinance|cash[- ]?out)\b", context))


def _signature(row, table_rows):
    """What makes this row *this* row regardless of wording: term, tier, flags."""
    label = normalize(row.label_text)
    span = _balance_range(row.label_text)
    cap = _loan_cap(re.sub(r"\$(\d{3})(\d{3})\b", r"$\1,\2", row.label_text))
    rank = None
    years = _model_years(row.label_text)
    if years:
        tiers = sorted({_model_years(r.label_text) for r in table_rows if _model_years(r.label_text)}, key=lambda s: -s[1])
        rank = tiers.index(years)
    flags = tuple(w for w in ("jumbo", "special", "promo", "introductory") if w in label)
    money = (round(span[0]), ) if span else (round(cap), ) if cap else ()
    if not row.terms and not money and rank is None and not flags:
        flags = (" ".join(sorted(set(tokens_of(row.label_text)) - {"rate", "account", "standard"})), )
    return (tuple(sorted(row.terms)), money, rank, flags)


def _table_words(rows):
    """The words that name a table (its headings), used to tell similar tables apart."""
    words = set()
    for row in rows:
        words |= set(tokens_of(f"{row.group} {row.product}"))
        if len(row.labels) > 1:
            words |= set(tokens_of(row.labels[0]))
    return words - {"rate", "account", "loan", "online", "only", "term"}


def _match_table(rows, current):
    """Which table of today's official rate sheet is this table a copy (or an older version) of?"""
    signatures = {_signature(r, rows) for r in rows}
    words = _table_words(rows)
    best, best_score, runner_up = None, 0.0, 0.0
    for key, official in current.items():
        target = {_signature(r, official) for r in official}
        union = signatures | target
        structure = len(signatures & target) / len(union) if union else 0.0
        if structure < 0.5:
            continue
        official_words = _table_words(official)
        both = words | official_words
        score = structure + (len(words & official_words) / len(both) if both else 0.0)
        if score > best_score:
            best, best_score, runner_up = key, score, best_score
        elif score > runner_up:
            runner_up = score
    if best is None or best_score - runner_up < 0.05:
        return None
    return best


def _value_text(row):
    """A row's values as text: "7.25% rate, 7.29% APR"."""
    return ", ".join(v if re.search(r"(?i)apr|apy", v) or m.lower() in v.lower() else f"{v} {m}" for m, v in row.values)


def _conflict_topic(row, table_rows):
    """The question a member would ask about this row, used to group rows into one conflict."""
    label = re.sub(r"(?i)\b(\d+)[\s-]*(month|year)s?\b", lambda m: f"{m.group(1)}-{m.group(2).lower()}", row.label_text)
    label = re.sub(r"\s+", " ", label.replace("–", "-")).strip()
    product = row.product or row.group
    if "mortgage" in row.tokens and row.terms:
        return f"{sorted(row.terms)[0][0]}-year fixed mortgage rate and APR (home loan, purchase)"
    years = _model_years(row.label_text)
    if years and "auto" in row.tokens:
        newest = max(_model_years(r.label_text)[1] for r in table_rows if _model_years(r.label_text))
        months = sorted(row.terms)[0][0] if row.terms else ""
        if years[1] == newest:
            return f"{months}-month auto loan APR ({row.labels[0]}, newest vehicles)"
        return "auto loan APR for older model years"
    if "certificate" in row.tokens:
        return f"{label} APY" if "certificate" in label.lower() else f"{label} certificate APY"
    metric = "APY" if row.value_of("APY") else "APR" if row.value_of("APR") else "rate"
    if metric.lower() in label.lower():
        metric = ""
    if label and label.lower() != product.lower():
        return (f"{label} {metric}" if product.lower() in label.lower() or not product else f"{product}: {label} {metric}").strip()
    return f"{product} {metric}".strip()


def _first_pct(text):
    match = re.search(r"(\d+(?:\.\d+)?)\s?%", text or "")
    return round(float(match.group(1)), 4) if match else None


def _metric_values(row):
    return {_first_pct(v) for _, v in row.values if _first_pct(v) is not None}


def _similar(a, b):
    """Two headings that share at least half their words."""
    a_words, b_words = set(tokens_of(a)), set(tokens_of(b))
    return bool(a_words and b_words) and len(a_words & b_words) / len(a_words | b_words) >= 0.5


def _norm_label(text):
    """A row label reduced to its essentials, so "$5,000.00 - $9,999" and "$5000-9999" compare equal."""
    text = normalize(text).replace("$", "").replace(",", "")
    text = re.sub(r"\.00\b", "", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\b(\d+) (month|year)s?\b", r"\1\2", text)
    return " ".join(sorted(set(re.findall(r"[a-z0-9.+-]+", text)) - {"account", "accounts", "rate", "rates", "the", "of", "or", "less", "-"}))


def _product_key(row):
    words = set(tokens_of(f"{row.product}")) - {"account", "rate", "loan", "online", "only"}
    return " ".join(sorted(words))


def _fact_key(row):
    """The same row on the rate sheet of another snapshot has the same product, label and audience."""
    return (frozenset(tokens_of(row.product or row.group)), _norm_label(row.label_text), row.business)


def _twin(row, rows):
    """The same row on another snapshot's rate sheet, if it exists."""
    key = _fact_key(row)
    return next((r for r in rows if r.tier == OFFICIAL and _fact_key(r) == key), None)


def _coverage(site, asked, answer_terms):
    """Share of the question's words (weighted by rarity) that the answer mentions.  A word the site
    never uses at all counts at full weight, since nothing could have answered it."""
    unseen = site.idf("\x00unseen")
    total = sum(site.idf(w) if w in site.postings else unseen for w in asked)
    return sum(site.idf(w) for w in asked if w in answer_terms) / total if total else 0.0


def _term_variants(terms):
    """{(5, "year")} -> also (60, "month"); {(24, "month")} -> also (2, "year")."""
    wanted = set()
    for n, unit in terms:
        wanted.add((n, unit))
        if unit == "year":
            wanted.add((n * 12, "month"))
        elif n % 12 == 0:
            wanted.add((n // 12, "year"))
    return wanted


def _stable_jumbo_last(rows):
    """Within each table, list the standard product before its jumbo sibling."""
    order, tables = [], defaultdict(list)
    for row in rows:
        key = (row.url, row.table)
        if key not in tables:
            order.append(key)
        tables[key].append(row)
    out = []
    for key in order:
        out.extend(sorted(tables[key], key=lambda r: "jumbo" in r.label_text.lower()))
    return out


def _dedupe_rows(rows):
    """Drop repeated rows (the same table appears on several pages)."""
    seen, out = set(), []
    for row in rows:
        key = (_norm_label(row.label_text), _product_key(row), tuple(v for _, v in row.values))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _describe(question):
    """A short description of what was asked, for "the website doesn't list ..." replies."""
    text = question.strip().rstrip("?.! ")
    text = re.sub(r"(?i)^(what(?:'s| is| are| was| were)?|can you tell me|tell me|do you (?:publish|have|offer)|how much (?:is|are|does|do)|get me|i would like to know|provide)\s+", "", text)
    text = re.sub(r"(?i)^(the |your |a |an )?(current )?((interest |dividend )?rates?|aprs?|apys?)\s+(do you charge\s+)?(on|for|of)\s+", "", text)
    return _trim(text.strip() or "that product", 80)


def _trim(text, limit):
    """Shorten text at a sentence or line break near the limit."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[:stop + 1] if stop > limit * 0.5 else cut.rsplit(" ", 1)[0]).rstrip() + (" ..." if stop <= limit * 0.5 else "")


def build():
    """Entry point the harness calls."""
    return KnowledgeSystem()


SearchKnowledge = KnowledgeSystem
