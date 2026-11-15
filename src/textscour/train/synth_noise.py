"""
Synthetic Noise Injection and Benchmark Generation Module for TEXTSCOUR.
Generates realistic promotional, CTA, boilerplate, handle, and URL spam noise
into clean text corpora with exact span attribution.
"""

from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Template:
    """A template for synthetic promotional noise injection."""
    lang: str
    kind: str
    template: str


@dataclass
class BenchDoc:
    """A benchmark document containing original text, noisy text, and ground truth spans."""
    original: str
    injected: str
    spans: list[tuple[int, int, str]]  # list of (start_char, end_char, kind)


# ============================================================================
# High-Quality Real Fillers (>= 100 each for Brands, Channels, URL Hosts)
# ============================================================================

BRANDS: list[str] = [
    "Amazon", "Flipkart", "Myntra", "Swiggy", "Zomato", "Nike", "Adidas", "Apple",
    "Samsung", "boAt", "Tata Neu", "Reliance Digital", "Nykaa", "Lenskart", "Puma",
    "Dell", "HP", "Lenovo", "Xiaomi", "OnePlus", "Realme", "Sony", "LG", "Canon",
    "Asus", "Acer", "Zara", "H&M", "Levi's", "Bata", "Titan", "Fastrack", "Casio",
    "Seiko", "Timex", "Mamaearth", "Sugar Cosmetics", "Lakmé", "Biotique", "WOW Skin",
    "Himalaya", "Dabur", "Patanjali", "ITC", "Nestlé", "Amul", "Parle", "Britannia",
    "Haldiram's", "Cadbury", "Pepsi", "Coca-Cola", "Red Bull", "Bisleri", "Aquafina",
    "Hyundai", "Maruti Suzuki", "Tata Motors", "Mahindra", "Toyota", "Honda", "Hero",
    "Bajaj Auto", "TVS Motors", "Royal Enfield", "Yamaha", "Ather Energy", "Ola Electric",
    "Uber", "Rapido", "MakeMyTrip", "Goibibo", "Cleartrip", "Yatra", "EaseMyTrip",
    "BookMyShow", "Paytm", "PhonePe", "Google Pay", "CRED", "Groww", "Zerodha",
    "Upstox", "Angel One", "ICICI Bank", "HDFC Bank", "SBI Card", "Axis Bank",
    "Kotak Mahindra", "Airtel", "Jio", "Vodafone Idea", "BSNL", "Infosys", "TCS",
    "Wipro", "HCL Tech", "Godrej", "Asian Paints", "Berger Paints", "Havells",
    "Crompton", "Philips", "Voltas", "Blue Star", "Daikin", "Whirlpool", "Bosch"
]

CHANNELS: list[str] = [
    "TechBurner", "Technical Guruji", "Geekyranjit", "T-Series", "CarryMinati",
    "BB Ki Vines", "Amit Bhadana", "Ashish Chanchlani", "Physics Wallah", "Unacademy",
    "Khan GS Research Centre", "Sandeep Maheshwari", "Vivek Bindra", "Ranveer Allahbadia",
    "The Ranveer Show", "DhiruMonchik", "Neon Man", "Gaurav Chaudhary", "Tanmay Bhat",
    "Samay Raina", "Chetan Bhagat Podcast", "Lallantop", "Aaj Tak", "NDTV India",
    "Zee News", "WION", "The Print", "Scroll", "Wire Hindi", "BBC Hindi",
    "Raj Shamani", "Finology Legal", "Pranjal Kamra", "CA Rachana Ranade", "Asset Yogi",
    "Labour Law Advisor", "Warikoo", "Ankur Warikoo", "Kunal Shah", "Sharan Hegde",
    "Finance With Sharan", "Warikoo Hindi", "BeerBiceps", "TRS Clips", "Raw and Real",
    "FilterCopy", "Dice Media", "TVF", "The Viral Fever", "Chai Sutta Chronicles",
    "Girliyapa", "Being Indian", "ScoopWhoop", "MensXP", "POPxo Hindi", "Pinkvilla",
    "Bollywood Hungama", "Film Companion", "Tried and Refused Productions", "Anupama Chopra",
    "Baradwaj Rangan", "Cinema Beyond Entertainment", "Comicverse", "Superheroes Talk",
    "BNFTV", "DesiNerd", "PJ Explained", "Men of Culture", "Bakra", "Khandesh Comedy",
    "Round2Hell", "R2H Vines", "Nazar Battu", "Hasley India", "Angry Prash",
    "BakLol Video", "Elvish Yadav Vlogs", "Fukra Insaan", "Triggered Insaan", "Mythpat",
    "Total Gaming", "Techno Gamerz", "A_S Gaming", "Lokesh Gamer", "Desi Gamers",
    "Gyan Gaming", "ScoutOP", "Mortal", "Jonathan Gaming", "Dynamo Gaming",
    "Payal Gaming", "Kronten Gaming", "Snax Gaming", "Mamba", "Thug Life Gaming",
    "Village Cooking Channel", "Grandpa Kitchen", "Kabitaskitchen", "Nisha Madhulika",
    "Ranveer Brar Official", "Sanjeev Kapoor Khazana", "BharatKaZayka", "Cook With Parul"
]

URL_HOSTS: list[str] = [
    "dealhub.in", "shopsmart.com", "lootdeals.org", "bit.ly", "tinyurl.com",
    "cutt.ly", "shorturl.at", "linktr.ee", "t.me", "wa.me",
    "discountking.in", "offerszone.net", "couponrani.com", "grabon.in", "coupondunia.in",
    "cashkaro.com", "desidime.com", "buybest.in", "megaoffers.online", "instadeals.store",
    "flashsaletoday.co", "savebigindia.com", "shoppingbazaar.in", "freestuff.in", "dealsplus.in",
    "hotdeals360.com", "couponzguru.com", "indialootdeals.com", "pricecheckindia.com", "smartprix.com",
    "mysmartprice.com", "91mobiles.com", "compareraja.in", "pricedekho.com", "jungleeoffers.com",
    "click2buy.in", "fastcheckout.site", "ecomdeals.biz", "promostore.live", "superdiscounts.club",
    "dailybogo.in", "lootalert.me", "teleoffers.xyz", "t.co", "is.gd",
    "rb.gy", "ow.ly", "buff.ly", "rebrand.ly", "soo.gd",
    "s.id", "v.gd", "clck.ru", "shorte.st", "adf.ly",
    "za.gl", "shrinkearn.com", "exe.io", "bc.vc", "ouo.io",
    "bestaffiliate.net", "referralslink.com", "clicktrack.io", "adtracker.info", "partnerpromo.org",
    "rewardsportal.in", "bonuspoints.store", "winprizes.club", "luckywheel.site", "giftcardsdaily.online",
    "cashbackhub.in", "paisaearn.co", "workfromhomeonline.biz", "dailyincome.cloud", "cryptoairdrop.win",
    "stocktipsfree.in", "tradingcalls.net", "jackpotdeals.info", "superbazaar.live", "bazaarindia.shop",
    "gadgetdeals.tech", "fashionbonanza.store", "grocerywholesale.in", "footwearfactory.net", "medicinediscount.in",
    "bookingexpress.club", "hoteldirectdeals.com", "flightcheaptickets.info", "tourpackagesindia.org", "homedecorsale.online",
    "bookmydeal.in", "festivesale2026.net", "newyearbonanza.com", "diwalisaleoffers.org", "independencedaydeals.in",
    "republicdaysale.com", "monsoonclearance.net", "summerbreezeoffers.store", "winterfashiondeals.biz", "endofseasonsale.in"
]

HANDLES: list[str] = [
    "@dealfinder", "@techguru", "@fashiondaily", "@loot_deals_india", "@gadgetworld",
    "@daily_offers", "@discount_king", "@smart_shopper", "@crypto_signals", "@stock_tips",
    "@fitness_freak", "@beauty_hacks", "@recipe_corner", "@travel_diaries", "@gaming_hub",
    "@meme_central", "@viral_clips", "@breaking_updates", "@quiz_daily", "@job_alerts",
    "@sarkari_result", "@current_affairs", "@motivational_quotes", "@fact_bytes", "@history_buff",
    "@science_nerd", "@coding_simplified", "@python_coder", "@ai_insights", "@startup_pulse"
]

PHONES: list[str] = [
    "+91 9876543210", "+91 9123456789", "+91 8765432109", "+91 7654321098",
    "+91 9988776655", "+91 9811223344", "+91 8800112233", "+91 7766554433",
    "1800-123-4567", "1800-200-9988", "1800-111-2222", "011-23456789",
    "+91-9871234560", "+91-8899001122", "+91 9311223344", "+91 9455667788"
]

CODES: list[str] = [
    "SAVE50", "DISCOUNT20", "FLAT30", "WELCOME10", "DIWALI2026",
    "MEGA70", "FREESHIP", "FESTIVE25", "BONUS100", "LUCKY40",
    "SUPERDEAL", "SUMMER15", "WINTER35", "CASHBACK50", "OFFER99"
]

PCTS: list[str] = [
    "10%", "15%", "20%", "25%", "30%", "40%", "50%", "60%", "70%", "75%", "80%"
]

PRICES: list[str] = [
    "₹199", "₹299", "₹499", "₹799", "₹999", "₹1,499", "₹1,999", "₹2,499",
    "₹4,999", "₹9,999", "Rs. 249", "Rs. 499", "Rs. 999", "Rs. 1,299", "$9.99", "$19.99"
]

APPS: list[str] = [
    "Flipkart", "Amazon Shopping", "Swiggy", "Zomato", "Meesho", "Myntra",
    "Paytm", "PhonePe", "CRED", "Zepto", "Blinkit", "Tata Neu", "Nykaa", "Ajio"
]


def load_templates(path: str) -> list[Template]:
    """
    Load synthetic noise templates from a TSV file.
    Expected format per line: lang<TAB>kind<TAB>template
    Lines starting with '#' or empty lines are ignored.
    """
    templates: list[Template] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            if len(row) < 3:
                continue
            lang = row[0].strip()
            kind = row[1].strip()
            template_text = row[2].strip()
            if template_text:
                templates.append(Template(lang=lang, kind=kind, template=template_text))

    return templates


def fill_template(tmpl_str: str, rng: random.Random, url_pool: Sequence[str] | None = None) -> str:
    """
    Replace placeholder slots with realistic synthetic entity values.
    Slots supported: {BRAND}, {HANDLE}, {URL}, {PHONE}, {CODE}, {PCT}, {PRICE}, {CHANNEL}, {APP}
    """
    res = tmpl_str
    if "{BRAND}" in res:
        res = res.replace("{BRAND}", rng.choice(BRANDS))
    if "{HANDLE}" in res:
        res = res.replace("{HANDLE}", rng.choice(HANDLES))
    if "{URL}" in res:
        host = rng.choice(url_pool) if url_pool else rng.choice(URL_HOSTS)
        protocol = rng.choice(["https://", "http://", "www."])
        path = rng.choice(["", "/deal", "/offer", f"/{rng.choice(CODES).lower()}", f"/r/{rng.randint(100, 999)}"])
        url = f"{protocol}{host}{path}"
        res = res.replace("{URL}", url)
    if "{PHONE}" in res:
        res = res.replace("{PHONE}", rng.choice(PHONES))
    if "{CODE}" in res:
        res = res.replace("{CODE}", rng.choice(CODES))
    if "{PCT}" in res:
        res = res.replace("{PCT}", rng.choice(PCTS))
    if "{PRICE}" in res:
        res = res.replace("{PRICE}", rng.choice(PRICES))
    if "{CHANNEL}" in res:
        res = res.replace("{CHANNEL}", rng.choice(CHANNELS))
    if "{APP}" in res:
        res = res.replace("{APP}", rng.choice(APPS))

    return res


_SENTENCE_SPLIT_REGEX = re.compile(r"([.!?।॥\n]+)")


def split_into_sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Splits text into sentence spans (start, end) preserving original text.
    Uses punctuation terminators and double newlines.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_SPLIT_REGEX.finditer(text):
        end = match.end()
        candidate = text[start:end].strip()
        if candidate:
            spans.append((start, end))
        start = end

    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))

    if not spans and text.strip():
        spans.append((0, len(text)))

    return spans


def inject(
    text: str,
    rng: random.Random | None = None,
    rate: float = 0.3,
    templates: Sequence[Template] | None = None,
    url_pool: Sequence[str] | None = None,
) -> tuple[str, list[tuple[int, int, str]]]:
    """
    Inject promotional sentences and inline tokens into text at sentence boundaries.

    Args:
        text: Original clean document text.
        rng: Optional Random instance for deterministic injection.
        rate: Probability of injecting a noise sentence at each sentence boundary.
        templates: Collection of Template objects to draw noise from.
        url_pool: Optional custom list of URL hosts.

    Returns:
        tuple (new_text, spans) where spans is a list of (start_char, end_char, kind).
    """
    if rng is None:
        rng = random.Random()

    if not templates:
        return text, []

    sentence_spans = split_into_sentence_spans(text)
    if not sentence_spans:
        return text, []

    injected_pieces: list[str] = []
    recorded_spans: list[tuple[int, int, str]] = []
    current_char_offset = 0

    for idx, (s_start, s_end) in enumerate(sentence_spans):
        orig_piece = text[s_start:s_end]
        injected_pieces.append(orig_piece)
        current_char_offset += len(orig_piece)

        # Decide whether to inject noise at this boundary
        # Ensure at least one injection if rate > 0 and this is the last piece and none injected yet
        should_inject = (rng.random() < rate) or (idx == len(sentence_spans) - 1 and not recorded_spans and rate > 0.05)

        if should_inject:
            template_obj = rng.choice(templates)
            filled_text = fill_template(template_obj.template, rng, url_pool)

            # Determine leading/trailing delimiter
            delim = " " if not orig_piece.endswith("\n") else ""
            noise_content = f"{delim}{filled_text}"

            span_start = current_char_offset + len(delim)
            span_end = span_start + len(filled_text)

            injected_pieces.append(noise_content)
            current_char_offset += len(noise_content)
            recorded_spans.append((span_start, span_end, template_obj.kind))

    new_text = "".join(injected_pieces)
    return new_text, recorded_spans


def make_benchmark(
    clean_docs: Sequence[str],
    templates_test: Sequence[Template],
    seed: int = 42,
) -> list[BenchDoc]:
    """
    Construct a deterministic synthetic benchmark suite by injecting held-out
    test templates into clean documents.

    Args:
        clean_docs: List of clean reference documents.
        templates_test: Test templates disjoint from training templates.
        seed: Random seed for repeatability.

    Returns:
        List of BenchDoc instances with ground-truth noise spans.
    """
    rng = random.Random(seed)
    bench_docs: list[BenchDoc] = []

    for doc in clean_docs:
        noisy_text, spans = inject(
            text=doc,
            rng=rng,
            rate=0.35,
            templates=templates_test,
        )
        bench_docs.append(BenchDoc(original=doc, injected=noisy_text, spans=spans))

    return bench_docs
