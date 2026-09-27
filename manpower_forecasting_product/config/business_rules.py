from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path(r"D:\Business planning")
MANPOWER_DIR = DEFAULT_BASE_DIR / "Manpower"
BUSINESS_DIR = DEFAULT_BASE_DIR / "Business"

MANPOWER_FY24_25_FILE = MANPOWER_DIR / "YTD Mar FY'24 & FY'25 - PDC & Productivity_V6.xlsx"
MANPOWER_FY26_FILE = MANPOWER_DIR / "IB Attrition Base File FY 25-26 31-03-26 Mar 26 YTD (1).xlsx"

BUSINESS_FILES = [
    {"fy": "FY24", "path": BUSINESS_DIR / "01_Accounting_IND_Apr'23 to Mar'24_YTD_Final_V1.xlsx"},
    {"fy": "FY25", "path": BUSINESS_DIR / "1_Accounting-Mar'25 YTD_IND_as on 31032025_V1.xlsx"},
    {"fy": "FY26", "path": BUSINESS_DIR / "1_Accounting-YTD March-26_IND_Final FY 26 (3).xlsx"},
]

CANONICAL_VINTAGES = ["0-3 M", "4-6 M", "7-12 M", "13-24 M", "24+ M"]
VALID_ZONES = {"East", "West", "North", "South"}

ZONE_ALIASES = {
    "east": "East", "e": "East", "east 1": "East", "east 2": "East", "east 3": "East", "east 4": "East",
    "west": "West", "w": "West", "west 1": "West", "west 2": "West", "west 3": "West", "west 4": "West",
    "north": "North", "n": "North", "north 1": "North", "north 2": "North", "north 3": "North", "north 4": "North",
    "south": "South", "s": "South", "south 1": "South", "south 2": "South", "south 3": "South", "south 4": "South",
    "kolkata": "East", "guwahati": "East", "siliguri": "East", "burdwan": "East", "patna": "East", "raipur": "East",
    "delhi": "North",
}

# FY26 Sub Channel / Business Vertical -> final Channel mapping.
CHANNEL_MAP = {
    "ib others": "IB Others",
    "policy bazaar": "Digital Partners",
    "jsfb": "JSFB",
    "axis bro/teller": "Axis Bank",
    "axis bro teller": "Axis Bank",
    "axis liability sales": "Axis Bank",
    "bandhan bank": "Bandhan Bank",
    "web online (aggregator)": "Digital Partners",
    "web online aggregator": "Digital Partners",
    "cabr": "CABR",
    "kvb": "KVB",
    "axis rm channel": "Axis Bank",
    "axis vertial center": "Axis Bank",
    "axis vertical center": "Axis Bank",
    "sib": "SIB",
    "shivalik sfb": "Shivalik",
    "jammu and kashmir bank": "JKB",
    "bfl": "BFL",
    "wealth management": "Wealth Management",
    "kbl": "KBL",
    "nr sales": "NR Sales",
    "rbl": "RBL",
    "yes bank ltd": "Yes Bank",
    "yes bank limited": "Yes Bank",
    "yes bank": "Yes Bank",
    "idfc": "IDFC",
    "usfb": "USFB",
    "equitas sfb": "Equitas SFB",
    "equitas bank": "Equitas SFB",
    "equitas": "Equitas SFB",
    "esaf": "ESAF",
    "dlb": "DLB",
    "dbs": "DBS",
    "rrb": "RRB",
    "cub": "CUB",
    "pcb": "PCB",
    "au sfb": "AU Bank",
    "axis bank others": "Axis Bank",
    "tmb": "TMB",
    "psb": "PSB",
    "federal bank": "Federal",
    "federal": "Federal",
    "axis white space": "Axis Bank",
    "others": "IB Others",
    "geojit": "Geojit",
    "doha": "Doha",
    "apgb": "APGB",
    "bfl-branch": "BFL",
    "bfl branch": "BFL",
    "bfl - branch": "BFL",
    "bfl fd": "BFL",
    "bfl - fd": "BFL",
    "bfl-fd": "BFL",
    "bfl wealth": "BFL",
    "bhfl": "BFL",
    "kvb_tn banca": "KVB",
    "kvb tn banca": "KVB",
    "cabr_tb": "CABR",
    "cabr tb": "CABR",
    "cabr_eb": "CABR",
    "cabr eb": "CABR",
    "equitas sfb_tn banca": "Equitas SFB",
    "equitas sfb tn banca": "Equitas SFB",
    "sfb_au": "AU Bank",
    "sfb au": "AU Bank",
    "sfb_usfb": "USFB",
    "sfb usfb": "USFB",
    "sfb_esaf": "ESAF",
    "sfb esaf": "ESAF",
    "cub_tn banca": "CUB",
    "cub tn banca": "CUB",
    "sfb_jsfb": "JSFB",
    "sfb jsfb": "JSFB",
    "tmb_tn banca": "TMB",
    "tmb tn banca": "TMB",
    "sfb_shivalik": "Shivalik",
    "sfb shivalik": "Shivalik",
    "sfb": "SFB",
    "axis bank retail": "Axis Bank",
    "axis bank - others": "Axis Bank",
}

EXCLUDE_CHANNEL_CONTAINS = ["credit life", "group"]
EXCLUDE_CHANNEL_EXACT = {"Ignore", "MFI Group", "NBFC"}

# Excel-VLOOKUP-equivalent ATS vintage breakpoints.
# TRUE/approximate match: largest breakpoint <= value.
ATS_VINTAGE_BREAKPOINTS = [
    (0.0, "0-3 M"),
    (4.0, "4-6 M"),
    (7.0, "7-12 M"),
    (13.0, "13-24 M"),
    (24.0001, "24+ M"),
]
