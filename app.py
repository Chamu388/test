import re
from typing import Tuple
from PIL import Image
import pytesseract
import pdfplumber
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
import uvicorn
import json
from datetime import datetime
from rapidfuzz import process, fuzz
import requests
from bs4 import BeautifulSoup
from fastapi.middleware.cors import CORSMiddleware
import traceback
import logging
from functools import lru_cache
from time import sleep
import time
import random

# --- SETUP OPENAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. Set it before starting the server.")
client = OpenAI(api_key=OPENAI_API_KEY)


app = FastAPI()

# Server logger (shows in Uvicorn/Railway logs)
logger = logging.getLogger("uvicorn.error")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Healthcheck ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Global exception handler for detailed logging ---
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("[UNHANDLED EXCEPTION]", tb)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# --- BANK & CUSTOMER DETECTION ---
def detect_bank_and_name(pdf_path: Path) -> Tuple[str, str]:
    customer_name = "Unknown"
    bank_name = "Unknown"
    KNOWN_BANKS = ["Monzo", "Barclays", "NatWest", "Lloyds", "TSB"]

    with pdfplumber.open(pdf_path) as pdf:
        first_page_text = ""
        first_page_words = []

        # Extract text from first 2 pages
        for page in pdf.pages[:2]:
            txt = extract_page_text(page)
            if txt:
                first_page_text += txt.lower()
            try:
                first_page_words.extend(page.extract_words())
            except:
                pass

        first_page = pdf.pages[0]
        page_width = first_page.width
        page_height = first_page.height

        # Convert first page to PIL image
        page_image = first_page.to_image(resolution=150).original

        # Crop top 10% for logo (faster and usually enough)
        logo_bbox = (0, 0, page_width, page_height * 0.10)
        logo_image = page_image.crop(logo_bbox)

        try:
            ocr_text = pytesseract.image_to_string(logo_image).lower()
        except Exception:
            ocr_text = ""

        # Detect bank from OCR first, then text
        for bank in KNOWN_BANKS:
            if bank.lower() in ocr_text or bank.lower() in first_page_text:
                bank_name = bank
                break

        # Fallback: URL or largest text
        if bank_name == "Unknown":
            url_match = re.search(r"www\.([a-z0-9\-]+)\.(co|com|uk)", first_page_text)
            if url_match:
                bank_name = url_match.group(1).capitalize()
            elif first_page_words:
                sorted_by_size = sorted(first_page_words, key=lambda w: w.get("size", 0), reverse=True)
                candidate = sorted_by_size[0]["text"]
                candidate = re.sub(r"[^A-Za-z\s&]", "", candidate).strip()
                if len(candidate) > 2:
                    bank_name = candidate

        # Extract customer name
        match = re.search(r"\b(Mr|Mrs|Ms|Dr)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", first_page_text, re.IGNORECASE)
        if match:
            customer_name = match.group(0)

    return bank_name, customer_name

def mask_personal_info(text: str) -> str:
    # Mask names
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", "[CUSTOMER_NAME]", text)
    # Mask account numbers (long sequences of digits)
    text = re.sub(r"\b\d{8,}\b", "[ACCOUNT_NUMBER]", text)
    # Mask sort codes or short numbers
    text = re.sub(r"\b\d{2}-\d{2}-\d{2}\b", "[SORT_CODE]", text)
    # Mask UK-style phone numbers
    text = re.sub(r"\b0\d{9,10}\b", "[PHONE_NUMBER]", text)
    # Mask addresses (simple heuristic: numbers + street names)
    text = re.sub(r"\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", "[ADDRESS]", text)
    return text


# --- Date Normalizer ---
def normalize_date(date_str: str) -> str:
    try:
        for fmt in ["%d %b %y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"]:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_str
    except Exception:
        return date_str


# --- Merchant List ---
MERCHANTS = [
    "NETFLIX.COM", "OVO ENERGY", "MARSHMALLOW INSURANCE",
    "COUNTRYWIDE RESIDE", "PLATINUM M/C", "YOUR-SAVING.COM", "KLARNA*TEMU.COM"
]

def smart_correct(desc: str) -> str:
    if not desc:
        return desc
    best, score, _ = process.extractOne(desc, MERCHANTS, scorer=fuzz.ratio)
    if score >= 80:
        print(f"[FUZZY] Corrected '{desc}' -> '{best}' (score {score})")
        return best
    return desc


# --- Page Extractor ---
def extract_page_text(page):
    words = page.extract_words(x_tolerance=2, y_tolerance=1, keep_blank_chars=True)
    if not words:
        return ""

    lines = {}
    for w in words:
        y = round(w["top"])
        lines.setdefault(y, []).append((w["x0"], w["text"]))

    result = []
    for y in sorted(lines.keys()):
        line = " ".join(t for _, t in sorted(lines[y], key=lambda x: x[0]))
        result.append(line)
    return "\n".join(result)


# --- GPT Parser (unchanged except no masking) ---
def gpt_extract_transactions(text: str, bank: str, first_page_text: str = "", ocr_text: str = ""):
    safe_text = mask_personal_info(text)
    logger.info("[GPT] Sending preprocessed text to model (first 1000 chars): %s", text[:1000])
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial data parser. Extract ONLY actual transactions. "
                        "Ignore headers, footers, account details, addresses, sort codes, and metadata. "
                        "Some words may be corrupted. Reconstruct merchant names but do NOT guess personal names. "
                        "Return JSON with 'bank' (string) and 'transactions': array of {date, raw_description, description, "
                        "amount, balance, currency}."
                    ),
                },
                {"role": "user", "content": f"Bank: {bank}\n\nTransaction lines:\n{text}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        logger.info("[GPT] Raw response (truncated 1000 chars): %s", str(raw_content)[:1000])

        parsed = json.loads(raw_content)

        # --- fix bank overwriting problem ---
        gpt_bank = parsed.get("bank", "Unknown")
        final_bank = gpt_bank if gpt_bank and gpt_bank.lower() not in ["unknown", "logo"] else bank

        # fallback: OCR detection if needed
        if final_bank.lower() in ["unknown", "logo"] and (first_page_text or ocr_text):
            detected_bank = detect_bank_and_name(first_page_text, ocr_text)
            if detected_bank != "Unknown":
                final_bank = detected_bank

        transactions = []
        prev_balance = None
        for tx in parsed.get("transactions", []):
            tx["date"] = normalize_date(tx.get("date", ""))

            try:
                amt = float(tx.get("amount", 0.0))
            except Exception:
                amt = 0.0
            try:
                tx["balance"] = float(tx["balance"])
            except Exception:
                tx["balance"] = 0.0

            tx["currency"] = "GBP"
            tx["bank"] = final_bank   # ✅ always use the fixed bank

            # classify money_in/out
            raw_desc = tx.get("raw_description", "").upper()
            money_in, money_out = None, None
            debit_keywords = ["BP", "CHG", "CHQ", "CPT", "DD", "DEB", "FEE", "FPO", "MPO", "PAY", "SO", "TFR"]
            credit_keywords = ["BGC", "DEP", "FPI", "MPI"]

            if any(k in raw_desc for k in debit_keywords):
                money_out = abs(amt)
            elif any(k in raw_desc for k in credit_keywords):
                money_in = abs(amt)
            else:
                if prev_balance is not None:
                    diff = round(tx["balance"] - prev_balance, 2)
                    if diff >= 0:
                        money_in = diff
                    else:
                        money_out = -diff

            tx["money_in"] = money_in
            tx["money_out"] = money_out

            tx.pop("amount", None)
            tx["description"] = smart_correct(tx.get("description", ""))

            if prev_balance is not None:
                expected = round(prev_balance + (money_in or 0) - (money_out or 0), 2)
                if abs(expected - tx["balance"]) > 0.05:
                    tx["note"] = f"⚠ Balance mismatch (expected {expected})"
            prev_balance = tx["balance"]

            transactions.append(tx)

        return transactions
    except Exception as e:
        logger.exception("[GPT] parse error: %s", e)
        return []


# --- Preprocess & Isolate functions unchanged ---
def preprocess_transactions(text: str) -> str:
    date_pattern = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+\w+\s+\d{2,4}\b")
    amount_pattern = re.compile(r"-?[\d,]+\.\d{2}")
    merged, buffer = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if date_pattern.search(line):
            if buffer:
                merged.append(buffer.strip())
            buffer = line
        elif amount_pattern.search(line):
            buffer += " " + line
        else:
            buffer += " " + line
    if buffer:
        merged.append(buffer.strip())
    cleaned = [row for row in merged if amount_pattern.search(row)]
    return "\n".join(cleaned)


def isolate_transactions(all_text: str, bank_name: str) -> str:
    lower_text = all_text.lower()
    if bank_name == "Lloyds":
        start_idx = lower_text.find("your transactions")
        if start_idx != -1:
            for marker in ["transaction types", "if you think", "lloyds bank plc", "registered office"]:
                end_idx = lower_text.find(marker, start_idx)
                if end_idx != -1:
                    return all_text[start_idx:end_idx]
            return all_text[start_idx:]
    return all_text

# --- Category / Subcategory Classification ---
CATEGORY_MAP = {
    "Essential Living Costs": {
        "Housing & Utilities": [
            "Rent / Mortgage payments",
            "Council Tax",
            "Electricity",
            "Gas",
            "Electricity and Gas",
            "Water",
            "Internet / Telephone / TV"
        ],
        "Food & Household": [
            "Groceries / Supermarkets",
            "Household goods (cleaning, toiletries, etc.)",
            "Dining / Takeaway / Restaurants"
        ]
    },

    "Transport & Travel": {
        "Fuel / Petrol / Diesel": [],
        "Public transport (bus, train, rail, underground)": [],
        "Taxis / Ride-hailing (Uber, PickMe, Bolt, etc.)": [],
        "Vehicle maintenance & insurance": [],
        "Parking fees / tolls": [],
        "Flights / Hotels / Holidays": []
    },

    "Family & Dependents": {
        "Childcare / Nursery / Babysitting": [],
        "School fees / Tuition": [],
        "Clothing & footwear": [],
        "Healthcare / Medicines / Insurance": [],
        "Elderly care / Family support payments": []
    },

    "Financial Commitments": {
        "Credit card payments": [],
        "Personal loans / HP agreements": [],
        "Overdraft fees": [],
        "Bank charges / Interest": [],
        "Bank transactions": ["Transfer out", "Cash withdrawal"],
        "Other mortgages / secured loans": []
    },

    "Lifestyle & Discretionary": {
        "Entertainment": [],
        "Subscriptions / Memberships": [],
        "Hobbies & Sports": [],
        "Retail": ["clothing and fashion", "general retail", "gifts, postage & stationery", "personal technology"],
        "Dining out / Coffee shops": [],
        "Travel & Holidays": []
    },

    "Income Categories": {
        "Salary (PAYE)": [],
        "Self-employment income": [],
        "Rental income": [],
        "Benefits / Government support": ["Universal Credit", "Child Benefit"],
        "Other recurring income": []
    }
}



# -----------------------
# FUNCTION: Free search helper (Wikipedia + hints)
# -----------------------

def search_wikipedia(company_name, max_results=2):
    """Free Wikipedia search for company info"""
    try:
        # Clean company name for Wikipedia search
        clean_name = re.sub(r'[^\w\s]', '', company_name).strip()
        if not clean_name:
            return []
        
        # Search Wikipedia API
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": f"{clean_name} company business",
            "srlimit": max_results
        }
        
        response = requests.get(search_url, params=params, timeout=5)
        if response.status_code != 200:
            return []
            
        data = response.json()
        results = []
        
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title and snippet:
                results.append(f"{title} - {snippet}")
                
        return results
    except Exception as e:
        logger.warning("[WIKI] Search error: %s", e)
        return []

@lru_cache(maxsize=256)
def search_vendor(shop_name, max_results=2):
    """Free search using Wikipedia + description hints"""
    shop_name = (shop_name or "").strip()
    if not shop_name:
        return []
    
    # Add context hints based on description
    q = shop_name.strip()
    q_l = q.lower()
    
    if "insurance" in q_l and "car" not in q_l and "motor" not in q_l:
        q += " car insurance"
    elif any(k in q_l for k in ["energy", "electric", "gas"]):
        q += " energy supplier"
    elif any(k in q_l for k in ["netflix", "prime", "spotify", "subscription"]):
        q += " streaming"
    
    logger.info("[SEARCH] query='%s'", q)
    
    # Try Wikipedia first (free, reliable)
    wiki_results = search_wikipedia(q, max_results)
    if wiki_results:
        return [f"QUERY: {q}"] + wiki_results
    
    # Fallback: return query with description-based hints
    hints = []
    if "insurance" in q_l:
        hints.append("insurance company")
    if "energy" in q_l:
        hints.append("energy supplier")
    if "netflix" in q_l or "streaming" in q_l:
        hints.append("streaming service")
    
    return [f"QUERY: {q}"] + hints


# -----------------------
# FUNCTION: Classify single transaction
# -----------------------
def classify_transaction(tx, model="gpt-3.5-turbo"):
    desc = tx.get("description", "")
    money_in = tx.get("money_in")
    money_out = tx.get("money_out")
    raw_desc = tx.get("raw_description", "")

    if not desc:
        tx["category"] = "Uncategorized"
        tx["subcategory"] = None
        tx["subsubcategory"] = None
        return tx

    # Perform vendor search first so we can force mappings based on clear signals
    search_results = search_vendor(desc, max_results=2)
    search_text = "\n".join(search_results) if search_results else "No search results found."
    combined_signal = f"{desc}\n{search_text}".lower()

    # FORCE: if car/vehicle/motor insurance is clearly indicated, classify accordingly (no matter what)
    if any(kw in combined_signal for kw in ["car insurance", "motor insurance", "vehicle insurance"]):
        tx["category"] = "Transport & Travel"
        tx["subcategory"] = "Vehicle maintenance & insurance"
        tx["subsubcategory"] = None
        tx["reasoning"] = "Search/description explicitly indicates car/motor/vehicle insurance, which maps to vehicle insurance."
        evidence_items = [desc[:80]]
        if search_results:
            evidence_items.append(search_results[0][:120])
        tx["evidence"] = evidence_items
        return tx

    # Heuristic: if looks like a personal name and it's a debit, treat as transfer out
    # Stricter detection to avoid brands/domains/cards being treated as names
    def is_probable_person_name(text: str) -> bool:
        candidate = (text or "").strip()
        # Quick fails: domains, slashes, digits, ampersands
        if any(ch in candidate for ch in ["/", "\\", ".", "@", "&", "_"]) or any(ch.isdigit() for ch in candidate):
            return False
        # Remove trailing transaction codes
        candidate = re.sub(r"\b(DD|DEB|CHQ|FPO|FPI|SO|TFR|BP)\b", " ", candidate, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z\s]", " ", candidate).strip()
        parts = [p for p in cleaned.split() if p]
        if len(parts) != 2:
            return False
        vendor_keywords = {
            "INSURANCE","ENERGY","NETFLIX","MASTERCARD","CARD","CREDIT","KLARNA","TEMU","COUNTRYWIDE","RESIDE",
            "SAVING","LIGHTCAST","DISCOVER","PLATINUM","YOUR","SAVINGS","BANK","PAYMENT","TRANSFER"
        }
        if any(p.upper() in vendor_keywords for p in parts):
            return False
        # Require Title Case for both names and alphabetic with vowels (to avoid acronyms)
        def looks_like_name(w: str) -> bool:
            return w[:1].isupper() and w[1:].islower() and any(v in w.lower() for v in "aeiou") and w.isalpha() and 2 <= len(w) <= 14
        if not all(looks_like_name(p) for p in parts):
            return False
        return True

    # Prefer classifying based on the cleaned merchant description (not raw);
    # only fall back to raw if description is empty
    name_text = desc or raw_desc
    if (money_out or 0) > 0 and is_probable_person_name(name_text):
        tx["category"] = "Financial Commitments"
        tx["subcategory"] = "Bank transactions"
        tx["subsubcategory"] = "Transfer out"
        tx["reasoning"] = "Description resembles a personal name; debit likely a person-to-person transfer."
        tx["evidence"] = [
            (name_text or "")[:80],
            "money_out>0",
            "name-like pattern detected"
        ]
        return tx

    # search_text already computed above

    prompt = f"""
You are an expert financial analyst. 
You must classify the following bank transaction into the most accurate category, subcategory, and sub-subcategory.
Use the CATEGORY_MAP exactly as given below for reference:

{json.dumps(CATEGORY_MAP, indent=2)}

Transaction description: "{desc}"
Money In: {money_in}, Money Out: {money_out}

Additional context from web search (quoted snippets may be truncated):
{search_text}

Instructions:
1. Think step by step and explain why this is the correct classification using both the description and any useful search snippets.
2. Then output the final labels.
3. If it is Money In, choose a category under "Income Categories".
4. If it is Money Out, choose a category under "Essential Living Costs", "Family & Dependents", "Financial Commitments", or "Lifestyle & Discretionary".
5. If the transaction mentions a person's name and it is Money Out, classify under "Financial Commitments" → "Transfer Out".
6. If search or description indicates vehicle/car/motor insurance (insurer brand names, policy, premium), classify under "Transport & Travel" → "Vehicle maintenance & insurance".
7. Respond ONLY in JSON (no extra text) with this shape:
{{
  "category": "CATEGORY",
  "subcategory": "SUBCATEGORY",
  "subsubcategory": "SUBSUBCATEGORY",
  "reasoning": "One short paragraph explaining your decision",
  "evidence": ["key term 1", "key term 2"]
}}
If a subcategory or sub-subcategory is not applicable, set it to null.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        tx["category"] = result.get("category", "Uncategorized")
        tx["subcategory"] = result.get("subcategory")
        tx["subsubcategory"] = result.get("subsubcategory")
        # include explanation fields for transparency
        if isinstance(result.get("reasoning"), str):
            tx["reasoning"] = result.get("reasoning")
        if isinstance(result.get("evidence"), list):
            tx["evidence"] = result.get("evidence")
        # defaults if model omitted fields
        if not isinstance(tx.get("reasoning"), str) or not tx["reasoning"].strip():
            tx["reasoning"] = "Model omitted reasoning; classification based on description and (if available) search context."
        if not isinstance(tx.get("evidence"), list) or len(tx["evidence"]) == 0:
            tx["evidence"] = [
                (desc or "")[0:80],
                f"money_in={money_in}",
                f"money_out={money_out}"
            ]
    except Exception as e:
        print("[Category Classification Error]", e)
        tx["category"] = "Uncategorized"
        tx["subcategory"] = None
        tx["subsubcategory"] = None

    return tx


def classify_all_transactions(transactions, model="gpt-3.5-turbo"):
    for i, tx in enumerate(transactions):
        transactions[i] = classify_transaction(tx, model=model)
    return transactions



# --- Extractor wrapper ---
def extract_transactions_from_pdf(pdf_path: Path, bank_name: str):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            txt = extract_page_text(page)
            if txt:
                all_text += txt + "\n"

    tx_block = isolate_transactions(all_text, bank_name)
    preprocessed = preprocess_transactions(tx_block)
    logger.info("[EXTRACT] chars_total=%d", len(all_text))
    first_lines = "\n".join(preprocessed.splitlines()[:20])
    logger.info("[EXTRACT] preprocessed_first_lines:\n%s", first_lines)
    transactions = gpt_extract_transactions(preprocessed, bank_name)
    transactions = classify_all_transactions(transactions)
    return transactions


# --- API Endpoint for multiple PDFs ---
@app.post("/extract-transactions")
async def extract_transactions(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        temp_path = Path(file.filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        bank, customer_name = detect_bank_and_name(temp_path)
        transactions = extract_transactions_from_pdf(temp_path, bank)
        temp_path.unlink()

        # --- FIX: override top-level bank with transaction bank if "logo"/"unknown" ---
        if transactions:
            tx_bank = transactions[0].get("bank")
            if bank.lower() in ["logo", "unknown"] and tx_bank and tx_bank.lower() not in ["logo", "unknown"]:
                bank = tx_bank

        results.append({
            "bank": bank,
            "customer_name": customer_name,
            "transactions": transactions
        })

    return JSONResponse(content={"results": results})



# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
