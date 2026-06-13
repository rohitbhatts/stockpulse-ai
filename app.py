import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="StockPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found in .env file. Please add it and restart.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==========================
# NSE/BSE STOCK UNIVERSE
# ==========================

@st.cache_data(ttl=86400)
def load_stock_universe():
    """Load complete NSE/BSE stock universe with sector mapping."""
    
    # Comprehensive NSE stock list with sectors
    # In production, this would come from NSE CSV download + database
    # This covers Nifty 500 + popular stocks across all sectors
    
    stocks = {
        # Banking
        "HDFCBANK": ("HDFC Bank", "Banking", "Private Bank"),
        "ICICIBANK": ("ICICI Bank", "Banking", "Private Bank"),
        "SBIN": ("State Bank of India", "Banking", "Public Bank"),
        "KOTAKBANK": ("Kotak Mahindra Bank", "Banking", "Private Bank"),
        "AXISBANK": ("Axis Bank", "Banking", "Private Bank"),
        "INDUSINDBK": ("IndusInd Bank", "Banking", "Private Bank"),
        "BANDHANBNK": ("Bandhan Bank", "Banking", "Private Bank"),
        "FEDERALBNK": ("Federal Bank", "Banking", "Private Bank"),
        "IDFCFIRSTB": ("IDFC First Bank", "Banking", "Private Bank"),
        "PNB": ("Punjab National Bank", "Banking", "Public Bank"),
        "BANKBARODA": ("Bank of Baroda", "Banking", "Public Bank"),
        "CANBK": ("Canara Bank", "Banking", "Public Bank"),
        "UNIONBANK": ("Union Bank of India", "Banking", "Public Bank"),
        "IOB": ("Indian Overseas Bank", "Banking", "Public Bank"),
        "RBLBANK": ("RBL Bank", "Banking", "Private Bank"),
        "YESBANK": ("Yes Bank", "Banking", "Private Bank"),
        "AUBANK": ("AU Small Finance Bank", "Banking", "Small Finance Bank"),
        "IDBI": ("IDBI Bank", "Banking", "Public Bank"),
        "MAHABANK": ("Bank of Maharashtra", "Banking", "Public Bank"),
        "INDIANB": ("Indian Bank", "Banking", "Public Bank"),

        # NBFC
        "BAJFINANCE": ("Bajaj Finance", "NBFC", "Consumer Finance"),
        "BAJAJFINSV": ("Bajaj Finserv", "NBFC", "Diversified Financial"),
        "SHRIRAMFIN": ("Shriram Finance", "NBFC", "Vehicle Finance"),
        "MUTHOOTFIN": ("Muthoot Finance", "NBFC", "Gold Finance"),
        "MANAPPURAM": ("Manappuram Finance", "NBFC", "Gold Finance"),
        "CHOLAFIN": ("Cholamandalam Finance", "NBFC", "Vehicle Finance"),
        "M&MFIN": ("Mahindra & Mahindra Finance", "NBFC", "Vehicle Finance"),
        "LICHSGFIN": ("LIC Housing Finance", "NBFC", "Housing Finance"),
        "PEL": ("Piramal Enterprises", "NBFC", "Diversified Financial"),
        "POONAWALLA": ("Poonawalla Fincorp", "NBFC", "Consumer Finance"),

        # Insurance
        "HDFCLIFE": ("HDFC Life Insurance", "Insurance", "Life Insurance"),
        "SBILIFE": ("SBI Life Insurance", "Insurance", "Life Insurance"),
        "ICICIPRULI": ("ICICI Prudential Life", "Insurance", "Life Insurance"),
        "LICI": ("Life Insurance Corporation", "Insurance", "Life Insurance"),
        "ICICIGI": ("ICICI Lombard General", "Insurance", "General Insurance"),
        "SBICARD": ("SBI Cards", "Insurance", "Credit Cards"),
        "STARHEALTH": ("Star Health Insurance", "Insurance", "Health Insurance"),
        "NIACL": ("New India Assurance", "Insurance", "General Insurance"),
        "GICRE": ("GIC Re", "Insurance", "Reinsurance"),

        # IT
        "TCS": ("Tata Consultancy Services", "IT", "IT Services"),
        "INFY": ("Infosys", "IT", "IT Services"),
        "WIPRO": ("Wipro", "IT", "IT Services"),
        "HCLTECH": ("HCL Technologies", "IT", "IT Services"),
        "TECHM": ("Tech Mahindra", "IT", "IT Services"),
        "LTIM": ("LTIMindtree", "IT", "IT Services"),
        "PERSISTENT": ("Persistent Systems", "IT", "IT Services"),
        "COFORGE": ("Coforge", "IT", "IT Services"),
        "MPHASIS": ("Mphasis", "IT", "IT Services"),
        "LTTS": ("L&T Technology Services", "IT", "Engineering R&D"),
        "TATAELXSI": ("Tata Elxsi", "IT", "Design & Technology"),
        "OFSS": ("Oracle Financial Services", "IT", "Fintech"),
        "NAUKRI": ("Info Edge (Naukri)", "IT", "Internet"),
        "ZOMATO": ("Zomato", "IT", "Internet"),
        "PAYTM": ("One97 Communications (Paytm)", "IT", "Fintech"),
        "POLICYBZR": ("PB Fintech (PolicyBazaar)", "IT", "Insurtech"),
        "DELHIVERY": ("Delhivery", "IT", "Logistics Tech"),
        "HAPPSTMNDS": ("Happiest Minds", "IT", "IT Services"),

        # Oil & Gas
        "RELIANCE": ("Reliance Industries", "Oil & Gas", "Diversified"),
        "ONGC": ("Oil & Natural Gas Corp", "Oil & Gas", "Exploration"),
        "IOC": ("Indian Oil Corporation", "Oil & Gas", "Refining"),
        "BPCL": ("Bharat Petroleum", "Oil & Gas", "Refining"),
        "HINDPETRO": ("Hindustan Petroleum", "Oil & Gas", "Refining"),
        "GAIL": ("GAIL India", "Oil & Gas", "Gas Distribution"),
        "PETRONET": ("Petronet LNG", "Oil & Gas", "LNG"),
        "OIL": ("Oil India", "Oil & Gas", "Exploration"),
        "ATGL": ("Adani Total Gas", "Oil & Gas", "City Gas Distribution"),
        "MGL": ("Mahanagar Gas", "Oil & Gas", "City Gas Distribution"),
        "IGL": ("Indraprastha Gas", "Oil & Gas", "City Gas Distribution"),
        "GUJGASLTD": ("Gujarat Gas", "Oil & Gas", "City Gas Distribution"),

        # Automobile
        "TATAMOTORS": ("Tata Motors", "Automobile", "Passenger Vehicle"),
        "MARUTI": ("Maruti Suzuki", "Automobile", "Passenger Vehicle"),
        "M&M": ("Mahindra & Mahindra", "Automobile", "SUV & Farm Equipment"),
        "BAJAJ-AUTO": ("Bajaj Auto", "Automobile", "Two Wheeler"),
        "HEROMOTOCO": ("Hero MotoCorp", "Automobile", "Two Wheeler"),
        "EICHERMOT": ("Eicher Motors", "Automobile", "Two Wheeler"),
        "ASHOKLEY": ("Ashok Leyland", "Automobile", "Commercial Vehicle"),
        "TVSMOTOR": ("TVS Motor", "Automobile", "Two Wheeler"),
        "TIINDIA": ("Tube Investments", "Automobile", "Auto Components"),
        "BALKRISIND": ("Balkrishna Industries", "Automobile", "Tyres"),
        "MRF": ("MRF", "Automobile", "Tyres"),
        "APOLLOTYRE": ("Apollo Tyres", "Automobile", "Tyres"),
        "BHARATFORG": ("Bharat Forge", "Automobile", "Auto Components"),
        "MOTHERSON": ("Samvardhana Motherson", "Automobile", "Auto Components"),
        "BOSCHLTD": ("Bosch", "Automobile", "Auto Components"),
        "EXIDEIND": ("Exide Industries", "Automobile", "Auto Batteries"),
        "OLECTRA": ("Olectra Greentech", "Automobile", "Electric Bus"),

        # Pharma & Healthcare
        "SUNPHARMA": ("Sun Pharma", "Pharma", "Pharma"),
        "DRREDDY": ("Dr. Reddy's Labs", "Pharma", "Pharma"),
        "CIPLA": ("Cipla", "Pharma", "Pharma"),
        "DIVISLAB": ("Divi's Laboratories", "Pharma", "API"),
        "LUPIN": ("Lupin", "Pharma", "Pharma"),
        "AUROPHARMA": ("Aurobindo Pharma", "Pharma", "Pharma"),
        "BIOCON": ("Biocon", "Pharma", "Biopharma"),
        "TORNTPHARM": ("Torrent Pharma", "Pharma", "Pharma"),
        "ALKEM": ("Alkem Laboratories", "Pharma", "Pharma"),
        "IPCALAB": ("IPCA Laboratories", "Pharma", "Pharma"),
        "LAURUSLABS": ("Laurus Labs", "Pharma", "API"),
        "GLENMARK": ("Glenmark Pharma", "Pharma", "Pharma"),
        "ZYDUSLIFE": ("Zydus Lifesciences", "Pharma", "Pharma"),
        "MANKIND": ("Mankind Pharma", "Pharma", "Pharma"),
        "APOLLOHOSP": ("Apollo Hospitals", "Healthcare", "Hospital"),
        "MAXHEALTH": ("Max Healthcare", "Healthcare", "Hospital"),
        "FORTIS": ("Fortis Healthcare", "Healthcare", "Hospital"),
        "METROPOLIS": ("Metropolis Healthcare", "Healthcare", "Diagnostics"),
        "LALPATHLAB": ("Dr. Lal PathLabs", "Healthcare", "Diagnostics"),

        # FMCG
        "HINDUNILVR": ("Hindustan Unilever", "FMCG", "Personal Care"),
        "ITC": ("ITC", "FMCG", "Diversified FMCG"),
        "NESTLEIND": ("Nestle India", "FMCG", "Food"),
        "BRITANNIA": ("Britannia Industries", "FMCG", "Food"),
        "TATACONSUM": ("Tata Consumer Products", "FMCG", "Food & Beverage"),
        "DABUR": ("Dabur India", "FMCG", "Personal Care"),
        "MARICO": ("Marico", "FMCG", "Personal Care"),
        "GODREJCP": ("Godrej Consumer Products", "FMCG", "Personal Care"),
        "COLPAL": ("Colgate-Palmolive India", "FMCG", "Oral Care"),
        "VBL": ("Varun Beverages", "FMCG", "Beverages"),
        "UBL": ("United Breweries", "FMCG", "Beverages"),
        "MCDOWELL-N": ("United Spirits", "FMCG", "Alcoholic Beverages"),
        "EMAMILTD": ("Emami", "FMCG", "Personal Care"),
        "BIKAJI": ("Bikaji Foods", "FMCG", "Snacks"),
        "PATANJALI": ("Patanjali Foods", "FMCG", "Food"),

        # Metals & Mining
        "TATASTEEL": ("Tata Steel", "Metals", "Steel"),
        "JSWSTEEL": ("JSW Steel", "Metals", "Steel"),
        "HINDALCO": ("Hindalco Industries", "Metals", "Aluminium"),
        "VEDL": ("Vedanta", "Metals", "Diversified Mining"),
        "COALINDIA": ("Coal India", "Metals", "Coal Mining"),
        "NMDC": ("NMDC", "Metals", "Iron Ore"),
        "SAIL": ("Steel Authority of India", "Metals", "Steel"),
        "NATIONALUM": ("National Aluminium", "Metals", "Aluminium"),
        "JINDALSTEL": ("Jindal Steel & Power", "Metals", "Steel"),
        "HINDZINC": ("Hindustan Zinc", "Metals", "Zinc"),
        "APLAPOLLO": ("APL Apollo Tubes", "Metals", "Steel Tubes"),
        "RATNAMANI": ("Ratnamani Metals", "Metals", "Steel Tubes"),

        # Power & Energy
        "NTPC": ("NTPC", "Power", "Thermal Power"),
        "POWERGRID": ("Power Grid Corp", "Power", "Transmission"),
        "ADANIGREEN": ("Adani Green Energy", "Power", "Renewable Energy"),
        "ADANIPOWER": ("Adani Power", "Power", "Thermal Power"),
        "TATAPOWER": ("Tata Power", "Power", "Diversified Power"),
        "NHPC": ("NHPC", "Power", "Hydro Power"),
        "SJVN": ("SJVN", "Power", "Hydro Power"),
        "IREDA": ("IREDA", "Power", "Power Finance"),
        "CESC": ("CESC", "Power", "Power Distribution"),
        "TORNTPOWER": ("Torrent Power", "Power", "Power Distribution"),
        "JSL": ("Jindal Stainless", "Power", "Stainless Steel"),

        # Chemicals
        "PIDILITIND": ("Pidilite Industries", "Chemicals", "Adhesives"),
        "SRF": ("SRF", "Chemicals", "Specialty Chemicals"),
        "ATUL": ("Atul", "Chemicals", "Specialty Chemicals"),
        "DEEPAKNTR": ("Deepak Nitrite", "Chemicals", "Specialty Chemicals"),
        "NAVINFLUOR": ("Navin Fluorine", "Chemicals", "Fluorochemicals"),
        "PIIND": ("PI Industries", "Chemicals", "Agrochemicals"),
        "UPL": ("UPL", "Chemicals", "Agrochemicals"),
        "AARTI": ("Aarti Industries", "Chemicals", "Specialty Chemicals"),
        "CLEAN": ("Clean Science", "Chemicals", "Specialty Chemicals"),
        "FLUOROCHEM": ("Gujarat Fluorochemicals", "Chemicals", "Fluorochemicals"),
        "ASIANPAINT": ("Asian Paints", "Chemicals", "Paints"),
        "BERGEPAINT": ("Berger Paints", "Chemicals", "Paints"),
        "KANSAINER": ("Kansai Nerolac", "Chemicals", "Paints"),

        # Capital Goods & Infrastructure
        "LT": ("Larsen & Toubro", "Capital Goods", "Engineering & Construction"),
        "SIEMENS": ("Siemens", "Capital Goods", "Engineering"),
        "ABB": ("ABB India", "Capital Goods", "Engineering"),
        "HAVELLS": ("Havells India", "Capital Goods", "Electrical Equipment"),
        "VOLTAS": ("Voltas", "Capital Goods", "AC & Cooling"),
        "BEL": ("Bharat Electronics", "Defence", "Defence Electronics"),
        "HAL": ("Hindustan Aeronautics", "Defence", "Aerospace"),
        "BHEL": ("Bharat Heavy Electricals", "Capital Goods", "Heavy Electrical"),
        "CUMMINSIND": ("Cummins India", "Capital Goods", "Engines"),
        "THERMAX": ("Thermax", "Capital Goods", "Energy Equipment"),
        "POLYCAB": ("Polycab India", "Capital Goods", "Cables & Wires"),
        "KEI": ("KEI Industries", "Capital Goods", "Cables & Wires"),
        "KAJARIACER": ("Kajaria Ceramics", "Capital Goods", "Building Materials"),
        "ASTRAZEN": ("AstraZeneca Pharma", "Pharma", "MNC Pharma"),

        # Cement
        "ULTRACEMCO": ("UltraTech Cement", "Cement", "Cement"),
        "SHREECEM": ("Shree Cement", "Cement", "Cement"),
        "AMBUJACEM": ("Ambuja Cements", "Cement", "Cement"),
        "ACC": ("ACC", "Cement", "Cement"),
        "DALMIACEM": ("Dalmia Bharat", "Cement", "Cement"),
        "RAMCOCEM": ("Ramco Cements", "Cement", "Cement"),
        "JKCEMENT": ("JK Cement", "Cement", "Cement"),
        "BIRLACORPN": ("Birla Corporation", "Cement", "Cement"),
        "NUVAMA": ("Nuvama Wealth", "NBFC", "Wealth Management"),

        # Real Estate
        "DLF": ("DLF", "Real Estate", "Real Estate"),
        "GODREJPROP": ("Godrej Properties", "Real Estate", "Real Estate"),
        "OBEROIRLTY": ("Oberoi Realty", "Real Estate", "Real Estate"),
        "PRESTIGE": ("Prestige Estates", "Real Estate", "Real Estate"),
        "LODHA": ("Macrotech Developers", "Real Estate", "Real Estate"),
        "PHOENIXLTD": ("Phoenix Mills", "Real Estate", "Retail REIT"),
        "BRIGADE": ("Brigade Enterprises", "Real Estate", "Real Estate"),
        "SOBHA": ("Sobha", "Real Estate", "Real Estate"),

        # Telecom
        "BHARTIARTL": ("Bharti Airtel", "Telecom", "Telecom Services"),
        "IDEA": ("Vodafone Idea", "Telecom", "Telecom Services"),
        "TTML": ("Tata Teleservices", "Telecom", "Telecom Services"),
        "INDUSTOWER": ("Indus Towers", "Telecom", "Telecom Infrastructure"),

        # Media & Entertainment
        "ZEEL": ("Zee Entertainment", "Media", "Broadcasting"),
        "PVR": ("PVR INOX", "Media", "Multiplex"),
        "SUNTV": ("Sun TV Network", "Media", "Broadcasting"),
        "NETWORK18": ("Network18 Media", "Media", "Media"),

        # Retail
        "DMART": ("Avenue Supermarts (DMart)", "Retail", "Supermarket"),
        "TRENT": ("Trent (Westside/Zudio)", "Retail", "Fashion Retail"),
        "TITAN": ("Titan Company", "Retail", "Jewellery & Watches"),
        "PAGEIND": ("Page Industries (Jockey)", "Retail", "Innerwear"),
        "ABFRL": ("Aditya Birla Fashion", "Retail", "Fashion Retail"),
        "SHOPERSTOP": ("Shoppers Stop", "Retail", "Department Store"),
        "RAYMOND": ("Raymond", "Retail", "Textiles & Fashion"),
        "DEVYANI": ("Devyani International", "Retail", "QSR"),
        "JUBLFOOD": ("Jubilant FoodWorks", "Retail", "QSR"),

        # Logistics
        "CONCOR": ("Container Corp of India", "Logistics", "Container Logistics"),
        "BLUEDART": ("Blue Dart Express", "Logistics", "Express Delivery"),
        "TCI": ("Transport Corp of India", "Logistics", "Multimodal Logistics"),

        # Textiles
        "ARVIND": ("Arvind", "Textiles", "Textiles"),
        "TRIDENT": ("Trident", "Textiles", "Home Textiles"),
        "WELSPUNLIV": ("Welspun Living", "Textiles", "Home Textiles"),
        "KPRMILL": ("KPR Mill", "Textiles", "Textiles"),

        # Adani Group
        "ADANIENT": ("Adani Enterprises", "Diversified", "Conglomerate"),
        "ADANIPORTS": ("Adani Ports & SEZ", "Infrastructure", "Ports"),
        "ADANIENERGY": ("Adani Energy Solutions", "Power", "Transmission"),
        "AWL": ("Adani Wilmar", "FMCG", "Edible Oil"),
        "ATGL": ("Adani Total Gas", "Oil & Gas", "City Gas Distribution"),

        # Miscellaneous Large Caps
        "IRCTC": ("IRCTC", "Tourism", "Railways & Tourism"),
        "INDIANHOTEL": ("Indian Hotels (Taj)", "Tourism", "Hotels"),
        "LEMONTRE": ("Lemon Tree Hotels", "Tourism", "Hotels"),
        "PIIND": ("PI Industries", "Chemicals", "Agrochemicals"),
        "JIOFIN": ("Jio Financial Services", "NBFC", "Financial Services"),
        "TATACOMM": ("Tata Communications", "Telecom", "Enterprise Telecom"),
        "TATATECH": ("Tata Technologies", "IT", "Engineering R&D"),
        "MOTILALOFS": ("Motilal Oswal Financial", "NBFC", "Broking"),
        "IIFL": ("IIFL Finance", "NBFC", "Diversified Financial"),
        "ANGELONE": ("Angel One", "NBFC", "Broking"),
        "BSE": ("BSE", "NBFC", "Stock Exchange"),
        "CDSL": ("CDSL", "NBFC", "Depository"),
        "MCX": ("Multi Commodity Exchange", "NBFC", "Commodity Exchange"),
        "CAMS": ("CAMS", "NBFC", "Mutual Fund Registrar"),
        "SONACOMS": ("Sona BLW Precision", "Automobile", "Auto Components"),
        "KAYNES": ("Kaynes Technology", "Capital Goods", "EMS"),
        "DIXON": ("Dixon Technologies", "Capital Goods", "EMS"),
        "AMBER": ("Amber Enterprises", "Capital Goods", "AC Components"),
        "COCHINSHIP": ("Cochin Shipyard", "Defence", "Shipbuilding"),
        "MAZAGON": ("Mazagon Dock Shipbuilders", "Defence", "Shipbuilding"),
        "GRINDWELL": ("Grindwell Norton", "Capital Goods", "Abrasives"),
        "CGPOWER": ("CG Power & Industrial", "Capital Goods", "Electrical Equipment"),
        "CROMPTON": ("Crompton Greaves CE", "Capital Goods", "Consumer Electricals"),
        "WHIRLPOOL": ("Whirlpool India", "Capital Goods", "Consumer Durables"),
        "BLUESTARLT": ("Blue Star", "Capital Goods", "AC & Cooling"),
        "SYRMA": ("Syrma SGS Technology", "Capital Goods", "EMS"),
        "JSWENERGY": ("JSW Energy", "Power", "Power Generation"),
        "RECLTD": ("REC", "Power", "Power Finance"),
        "PFC": ("Power Finance Corp", "Power", "Power Finance"),
        "HUDCO": ("HUDCO", "Infrastructure", "Housing Finance"),
        "RVNL": ("Rail Vikas Nigam", "Infrastructure", "Railway Construction"),
        "IRFC": ("Indian Railway Finance", "Infrastructure", "Railway Finance"),
        "NBCC": ("NBCC India", "Infrastructure", "Construction"),
        "NCC": ("NCC", "Infrastructure", "Construction"),
        "KEC": ("KEC International", "Infrastructure", "Power T&D"),
        "KALPATPOWR": ("Kalpataru Projects", "Infrastructure", "Power T&D"),
    }

    # Build DataFrame from hardcoded stocks (fallback)
    rows = []
    for symbol, (name, sector, industry) in stocks.items():
        rows.append({
            "symbol": symbol,
            "company_name": name,
            "sector": sector,
            "industry": industry,
            "search_text": f"{symbol} {name} {sector} {industry}".lower()
        })

    hardcoded_df = pd.DataFrame(rows)
    hardcoded_symbols = set(hardcoded_df["symbol"].tolist())

    # Try to fetch NSE equity list for 2000+ stocks
    try:
        import urllib.request
        import io
        import csv

        nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        req = urllib.request.Request(nse_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/csv"
        })
        response = urllib.request.urlopen(req, timeout=10)
        csv_data = response.read().decode("utf-8")
        
        nse_reader = csv.DictReader(io.StringIO(csv_data))
        nse_rows = []
        for r in nse_reader:
            sym = r.get("SYMBOL", "").strip()
            name = r.get("NAME OF COMPANY", "").strip()
            if sym and name and sym not in hardcoded_symbols:
                # Auto-assign sector as "Other" for stocks not in our mapping
                nse_rows.append({
                    "symbol": sym,
                    "company_name": name,
                    "sector": "Other",
                    "industry": "Equity",
                    "search_text": f"{sym} {name} Other Equity".lower()
                })
        
        if nse_rows:
            nse_df = pd.DataFrame(nse_rows)
            combined_df = pd.concat([hardcoded_df, nse_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset="symbol", keep="first")
            return combined_df
    except Exception:
        pass  # Fallback to hardcoded

    return hardcoded_df


def search_stocks(query, stock_df, max_results=8):
    """Search stocks by symbol, company name, sector, or industry."""
    if not query or len(query) < 1:
        return stock_df.head(max_results)

    query_lower = query.lower().strip()

    # Exact symbol match first
    exact = stock_df[stock_df["symbol"].str.lower() == query_lower]
    if not exact.empty:
        return exact

    # Contains match
    matches = stock_df[stock_df["search_text"].str.contains(query_lower, na=False)]
    
    # Sort: symbol starts-with first, then name starts-with, then rest
    if not matches.empty:
        matches = matches.copy()
        matches["sort_score"] = 0
        matches.loc[matches["symbol"].str.lower().str.startswith(query_lower), "sort_score"] = 2
        matches.loc[matches["company_name"].str.lower().str.startswith(query_lower), "sort_score"] = 1
        matches = matches.sort_values("sort_score", ascending=False).drop("sort_score", axis=1)
    
    return matches.head(max_results)


# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background-color: #0a0e17;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f1629 0%, #1a1f3a 50%, #0f1629 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, #a855f7, transparent);
    }
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        background: linear-gradient(135deg, #e2e8f0 0%, #a855f7 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.03em;
    }
    .main-header p {
        color: #64748b;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    .metric-card {
        background: linear-gradient(145deg, #111827 0%, #0f1629 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-delta-up { color: #22c55e; font-size: 0.85rem; font-weight: 600; }
    .metric-delta-down { color: #ef4444; font-size: 0.85rem; font-weight: 600; }

    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
    }
    .score-strong-buy { background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4); color: #22c55e; }
    .score-buy { background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); color: #4ade80; }
    .score-hold { background: rgba(234, 179, 8, 0.1); border: 1px solid rgba(234, 179, 8, 0.3); color: #eab308; }
    .score-sell { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #ef4444; }

    .section-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.15);
        letter-spacing: -0.01em;
    }

    .indicator-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .indicator-name { font-size: 0.8rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .indicator-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }
    .signal-bullish { color: #22c55e; font-weight: 600; font-size: 0.8rem; }
    .signal-bearish { color: #ef4444; font-weight: 600; font-size: 0.8rem; }
    .signal-neutral { color: #eab308; font-weight: 600; font-size: 0.8rem; }

    .ai-analysis-box {
        background: linear-gradient(145deg, #111827 0%, #0f1629 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.7;
        font-size: 0.92rem;
        color: #cbd5e1;
    }
    .ai-analysis-box::before {
        content: '';
        display: block;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #a855f7, transparent);
        margin: -1.5rem -1.5rem 1.2rem -1.5rem;
        border-radius: 12px 12px 0 0;
    }

    /* AI Report Card */
    .ai-report {
        background: linear-gradient(145deg, #111827 0%, #0f1629 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .ai-report-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .ai-report-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .signal-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .signal-check { color: #22c55e; }
    .signal-cross { color: #ef4444; }
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .risk-low { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .risk-medium { background: rgba(234, 179, 8, 0.15); color: #eab308; }
    .risk-high { background: rgba(239, 68, 68, 0.15); color: #ef4444; }

    /* Search result styling */
    .search-result {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.3rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .search-result:hover {
        border-color: rgba(99, 102, 241, 0.3);
        background: #151d30;
    }

    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid rgba(99, 102, 241, 0.1);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stPlotlyChart { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 0.5rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #111827; border-radius: 10px; padding: 4px; border: 1px solid rgba(255,255,255,0.06); }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-weight: 500; font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { background: rgba(99, 102, 241, 0.2) !important; color: #e2e8f0 !important; }

    /* ===== MOBILE RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-header { padding: 1.2rem 1rem; }
        .main-header h1 { font-size: 1.4rem; }
        .main-header p { font-size: 0.8rem; }

        .metric-card { padding: 0.8rem; }
        .metric-value { font-size: 1.1rem; }
        .metric-label { font-size: 0.65rem; }

        .section-header { font-size: 1rem; }

        .indicator-card { padding: 0.7rem 0.9rem; }
        .indicator-value { font-size: 0.95rem; }

        .ai-analysis-box { padding: 1rem; font-size: 0.85rem; }

        .score-badge { padding: 0.4rem 1rem; font-size: 0.85rem; }

        .stTabs [data-baseweb="tab"] { font-size: 0.7rem; padding: 0.4rem 0.5rem; }

        /* Force columns to stack on mobile */
        [data-testid="column"] { width: 100% !important; flex: 100% !important; min-width: 100% !important; }
    }

    @media (max-width: 480px) {
        .main-header h1 { font-size: 1.2rem; }
        .metric-value { font-size: 1rem; }
        .metric-label { font-size: 0.6rem; }
    }
</style>
""", unsafe_allow_html=True)


# ==========================
# HELPER FUNCTIONS
# ==========================

@st.cache_data(ttl=300)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        return None
    return data


def flatten_col(series):
    """Flatten multi-index column if present."""
    if hasattr(series, "columns"):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def compute_indicators(data):
    """Compute all technical indicators."""
    clean_data = data.dropna()

    close = flatten_col(clean_data["Close"])
    high = flatten_col(clean_data["High"])
    low = flatten_col(clean_data["Low"])
    volume = flatten_col(clean_data["Volume"])

    indicators = {}

    indicators["close"] = close
    indicators["high"] = high
    indicators["low"] = low
    indicators["volume"] = volume
    indicators["current_price"] = float(close.iloc[-1])

    prev_close = float(close.iloc[-2]) if len(close) > 1 else indicators["current_price"]
    indicators["price_change"] = indicators["current_price"] - prev_close
    indicators["price_change_pct"] = (indicators["price_change"] / prev_close) * 100

    for w in [9, 20, 50, 100, 200]:
        ema = EMAIndicator(close, window=w).ema_indicator()
        indicators[f"ema{w}"] = ema
        indicators[f"latest_ema{w}"] = float(ema.dropna().iloc[-1]) if len(ema.dropna()) > 0 else None

    rsi = RSIIndicator(close, window=14).rsi()
    indicators["rsi"] = rsi
    indicators["latest_rsi"] = float(rsi.dropna().iloc[-1]) if len(rsi.dropna()) > 0 else None

    macd_obj = MACD(close)
    indicators["macd"] = macd_obj.macd()
    indicators["macd_signal"] = macd_obj.macd_signal()
    indicators["macd_histogram"] = macd_obj.macd_diff()
    indicators["latest_macd"] = float(indicators["macd"].dropna().iloc[-1]) if len(indicators["macd"].dropna()) > 0 else None
    indicators["latest_macd_signal"] = float(indicators["macd_signal"].dropna().iloc[-1]) if len(indicators["macd_signal"].dropna()) > 0 else None

    bb = BollingerBands(close, window=20, window_dev=2)
    indicators["bb_upper"] = bb.bollinger_hband()
    indicators["bb_middle"] = bb.bollinger_mavg()
    indicators["bb_lower"] = bb.bollinger_lband()
    indicators["latest_bb_upper"] = float(indicators["bb_upper"].dropna().iloc[-1]) if len(indicators["bb_upper"].dropna()) > 0 else None
    indicators["latest_bb_lower"] = float(indicators["bb_lower"].dropna().iloc[-1]) if len(indicators["bb_lower"].dropna()) > 0 else None

    adx = ADXIndicator(high, low, close, window=14)
    indicators["adx"] = adx.adx()
    indicators["latest_adx"] = float(indicators["adx"].dropna().iloc[-1]) if len(indicators["adx"].dropna()) > 0 else None

    atr = AverageTrueRange(high, low, close, window=14)
    indicators["atr"] = atr.average_true_range()
    indicators["latest_atr"] = float(indicators["atr"].dropna().iloc[-1]) if len(indicators["atr"].dropna()) > 0 else None

    stoch = StochasticOscillator(high, low, close)
    indicators["stoch_k"] = stoch.stoch()
    indicators["stoch_d"] = stoch.stoch_signal()
    indicators["latest_stoch_k"] = float(indicators["stoch_k"].dropna().iloc[-1]) if len(indicators["stoch_k"].dropna()) > 0 else None

    obv = OnBalanceVolumeIndicator(close, volume)
    indicators["obv"] = obv.on_balance_volume()

    recent_high = float(high.tail(20).max())
    recent_low = float(low.tail(20).min())
    pivot = (recent_high + recent_low + indicators["current_price"]) / 3
    indicators["pivot"] = pivot
    indicators["support_1"] = 2 * pivot - recent_high
    indicators["resistance_1"] = 2 * pivot - recent_low
    indicators["support_2"] = pivot - (recent_high - recent_low)
    indicators["resistance_2"] = pivot + (recent_high - recent_low)

    indicators["week_52_high"] = float(high.max())
    indicators["week_52_low"] = float(low.min())

    return indicators


@st.cache_data(ttl=600)
def get_fundamentals(ticker):
    """Fetch fundamental data for a stock."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if not info or len(info) < 5:
            return None
        return info
    except Exception:
        return None


def compute_score(ind):
    """Compute technical score out of 100."""
    score = 0
    reasons = []

    rsi = ind.get("latest_rsi")
    if rsi:
        if 40 <= rsi <= 60:
            score += 10
            reasons.append(("RSI neutral zone", "neutral"))
        elif rsi > 60:
            score += 20
            reasons.append(("RSI bullish momentum", "bullish"))
        elif rsi < 30:
            score += 15
            reasons.append(("RSI oversold — potential bounce", "neutral"))
        else:
            score += 5
            reasons.append(("RSI weak momentum", "bearish"))

    price = ind["current_price"]
    ema20 = ind.get("latest_ema20")
    ema50 = ind.get("latest_ema50")
    ema200 = ind.get("latest_ema200")

    ema_count = 0
    if ema20 and price > ema20: ema_count += 1
    if ema50 and price > ema50: ema_count += 1
    if ema200 and price > ema200: ema_count += 1

    ema_score = int((ema_count / 3) * 25)
    score += ema_score
    if ema_count == 3: reasons.append(("Price above all key EMAs", "bullish"))
    elif ema_count >= 2: reasons.append(("Price above most EMAs", "bullish"))
    elif ema_count == 1: reasons.append(("Price above only short-term EMA", "neutral"))
    else: reasons.append(("Price below all EMAs", "bearish"))

    macd_val = ind.get("latest_macd")
    macd_sig = ind.get("latest_macd_signal")
    if macd_val is not None and macd_sig is not None:
        if macd_val > macd_sig:
            score += 20
            reasons.append(("MACD bullish crossover", "bullish"))
        elif macd_val > 0:
            score += 10
            reasons.append(("MACD positive but weakening", "neutral"))
        else:
            reasons.append(("MACD bearish", "bearish"))

    bb_upper = ind.get("latest_bb_upper")
    bb_lower = ind.get("latest_bb_lower")
    if bb_upper and bb_lower:
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pos = (price - bb_lower) / bb_range
            if 0.3 <= bb_pos <= 0.7:
                score += 15
                reasons.append(("Price in healthy BB zone", "bullish"))
            elif bb_pos > 0.7:
                score += 8
                reasons.append(("Price near upper BB — overbought risk", "neutral"))
            else:
                score += 5
                reasons.append(("Price near lower BB — oversold", "neutral"))

    adx = ind.get("latest_adx")
    if adx:
        if adx > 25:
            score += 10
            reasons.append(("Strong trend (ADX > 25)", "bullish"))
        elif adx > 20:
            score += 5
            reasons.append(("Moderate trend", "neutral"))
        else:
            score += 2
            reasons.append(("Weak/No trend", "bearish"))

    volume = ind.get("volume")
    if volume is not None and len(volume) > 20:
        avg_vol = float(volume.tail(20).mean())
        latest_vol = float(volume.iloc[-1])
        if latest_vol > avg_vol * 1.2:
            score += 10
            reasons.append(("Volume above average — confirms move", "bullish"))
        elif latest_vol > avg_vol * 0.8:
            score += 5
            reasons.append(("Volume near average", "neutral"))
        else:
            score += 2
            reasons.append(("Low volume — weak conviction", "bearish"))

    score = min(score, 100)

    if score >= 75: recommendation, rec_class = "STRONG BUY", "score-strong-buy"
    elif score >= 55: recommendation, rec_class = "BUY", "score-buy"
    elif score >= 35: recommendation, rec_class = "HOLD", "score-hold"
    else: recommendation, rec_class = "SELL", "score-sell"

    return score, recommendation, rec_class, reasons


def build_ai_report(ind, score, recommendation, reasons, fund_info=None):
    """Build structured AI research report (non-GPT, instant)."""
    report = {}

    # Trend
    price = ind["current_price"]
    ema20 = ind.get("latest_ema20", 0)
    ema50 = ind.get("latest_ema50", 0)
    ema200 = ind.get("latest_ema200", 0)

    if price > ema20 and price > ema50 and price > ema200:
        report["trend"] = ("Bullish", "#22c55e")
    elif price < ema20 and price < ema50:
        report["trend"] = ("Bearish", "#ef4444")
    else:
        report["trend"] = ("Sideways", "#eab308")

    # Technical Signals
    tech_signals = []
    rsi = ind.get("latest_rsi", 50)
    tech_signals.append(("RSI Strong" if rsi > 55 else "RSI Weak" if rsi < 45 else "RSI Neutral", rsi > 50))
    
    macd_v = ind.get("latest_macd", 0)
    macd_s = ind.get("latest_macd_signal", 0)
    tech_signals.append(("MACD Positive" if macd_v > macd_s else "MACD Negative", macd_v > macd_s))
    
    tech_signals.append((f"Price {'Above' if price > ema20 else 'Below'} EMA20", price > ema20))
    tech_signals.append((f"Price {'Above' if price > ema50 else 'Below'} EMA50", price > ema50))
    tech_signals.append((f"Price {'Above' if price > ema200 else 'Below'} EMA200", price > ema200))
    
    adx = ind.get("latest_adx", 0)
    tech_signals.append(("Strong Trend" if adx > 25 else "Weak Trend", adx > 25))
    report["tech_signals"] = tech_signals

    # Fundamental Signals
    fund_signals = []
    if fund_info:
        pe = fund_info.get("trailingPE")
        if pe: fund_signals.append((f"P/E: {pe:.1f}" + (" (Attractive)" if pe < 25 else " (Elevated)"), pe < 25))
        
        roe = fund_info.get("returnOnEquity")
        if roe: fund_signals.append((f"ROE: {roe*100:.1f}%" + (" (Strong)" if roe > 0.15 else " (Weak)"), roe > 0.15))
        
        de = fund_info.get("debtToEquity")
        if de: fund_signals.append((f"D/E: {de:.1f}" + (" (Low)" if de < 100 else " (High)"), de < 100))
        
        rg = fund_info.get("revenueGrowth")
        if rg: fund_signals.append((f"Revenue Growth: {rg*100:.1f}%", rg > 0.05))
        
        pm = fund_info.get("profitMargins")
        if pm: fund_signals.append((f"Profit Margin: {pm*100:.1f}%", pm > 0.10))
        
        dy = fund_info.get("dividendYield")
        if dy: fund_signals.append((f"Dividend Yield: {dy*100:.2f}%", dy > 0.01))
    report["fund_signals"] = fund_signals

    # Risk Level
    bearish_count = sum(1 for _, s in reasons if s == "bearish")
    if bearish_count >= 3: report["risk"] = ("High", "risk-high")
    elif bearish_count >= 1: report["risk"] = ("Medium", "risk-medium")
    else: report["risk"] = ("Low", "risk-low")

    # Strategy
    if score >= 75:
        report["strategy"] = f"Strong uptrend confirmed. Consider buying on dips near ₹{ind.get('support_1', 0):.2f}. Target: ₹{ind.get('resistance_1', 0):.2f}."
    elif score >= 55:
        report["strategy"] = f"Moderate bullish setup. Entry near ₹{ind.get('support_1', 0):.2f} with stop-loss at ₹{ind.get('support_2', 0):.2f}."
    elif score >= 35:
        report["strategy"] = f"Sideways market. Wait for breakout above ₹{ind.get('resistance_1', 0):.2f} or breakdown below ₹{ind.get('support_1', 0):.2f}."
    else:
        report["strategy"] = f"Bearish trend. Avoid fresh entries. Watch for reversal above ₹{float(ind.get('latest_ema20', 0)):.2f} (EMA20)." if isinstance(ind.get('latest_ema20'), (int, float)) else "Bearish trend. Avoid fresh entries."

    return report


def create_main_chart(ind, stock_name):
    """Create professional multi-panel chart."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.17, 0.17, 0.16]
    )

    close = ind["close"]

    fig.add_trace(go.Scatter(
        x=close.index, y=close, name="Price",
        line=dict(color="#6366f1", width=2),
        fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.06)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ind["bb_upper"].index, y=ind["bb_upper"], name="BB Upper",
        line=dict(color="rgba(168, 85, 247, 0.3)", width=1, dash="dot"), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ind["bb_lower"].index, y=ind["bb_lower"], name="BB Lower",
        line=dict(color="rgba(168, 85, 247, 0.3)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(168, 85, 247, 0.04)", showlegend=False
    ), row=1, col=1)

    ema_colors = {"ema20": "#22c55e", "ema50": "#eab308", "ema200": "#ef4444"}
    for ema_name, color in ema_colors.items():
        ema_data = ind[ema_name]
        fig.add_trace(go.Scatter(
            x=ema_data.index, y=ema_data, name=ema_name.upper(),
            line=dict(color=color, width=1.2)
        ), row=1, col=1)

    fig.add_hline(y=ind["resistance_1"], line_dash="dash", line_color="rgba(239, 68, 68, 0.4)", annotation_text="R1", row=1, col=1)
    fig.add_hline(y=ind["support_1"], line_dash="dash", line_color="rgba(34, 197, 94, 0.4)", annotation_text="S1", row=1, col=1)

    colors = ["#22c55e" if close.iloc[i] >= close.iloc[i-1] else "#ef4444" for i in range(1, len(close))]
    colors.insert(0, "#64748b")
    fig.add_trace(go.Bar(
        x=ind["volume"].index, y=ind["volume"], name="Volume",
        marker_color=colors, opacity=0.5, showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=ind["rsi"].index, y=ind["rsi"], name="RSI",
        line=dict(color="#a855f7", width=1.5)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.3)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(34, 197, 94, 0.3)", row=3, col=1)

    fig.add_trace(go.Scatter(x=ind["macd"].index, y=ind["macd"], name="MACD", line=dict(color="#6366f1", width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=ind["macd_signal"].index, y=ind["macd_signal"], name="Signal", line=dict(color="#ef4444", width=1.2)), row=4, col=1)
    hist = ind["macd_histogram"]
    hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in hist]
    fig.add_trace(go.Bar(x=hist.index, y=hist, name="Histogram", marker_color=hist_colors, opacity=0.4, showlegend=False), row=4, col=1)

    fig.update_layout(
        height=800, template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0f1629",
        font=dict(family="Inter", color="#94a3b8", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=30, b=30), xaxis_rangeslider_visible=False, hovermode="x unified"
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1, gridcolor="rgba(255,255,255,0.03)")
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="rgba(255,255,255,0.03)")
    fig.update_yaxes(title_text="RSI", row=3, col=1, gridcolor="rgba(255,255,255,0.03)")
    fig.update_yaxes(title_text="MACD", row=4, col=1, gridcolor="rgba(255,255,255,0.03)")
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)", row=i, col=1)

    return fig


def get_ai_analysis(stock, ind, score, recommendation, fundamentals=None):
    """Get AI analysis from OpenAI."""
    fund_section = ""
    if fundamentals:
        fund_section = f"""
Fundamental Data:
- Market Cap: {fundamentals.get('mcap', 'N/A')}
- P/E (TTM): {fundamentals.get('pe', 'N/A')}
- EPS: {fundamentals.get('eps', 'N/A')}
- ROE: {fundamentals.get('roe', 'N/A')}
- Debt/Equity: {fundamentals.get('de', 'N/A')}
- Revenue Growth: {fundamentals.get('rev_growth', 'N/A')}
- Profit Margin: {fundamentals.get('margin', 'N/A')}
- Dividend Yield: {fundamentals.get('div_yield', 'N/A')}
"""

    prompt = f"""You are an expert stock market analyst. Analyze this NSE stock with precision.

Stock: {stock}
Current Price: ₹{ind['current_price']:.2f}
Price Change: {ind['price_change_pct']:.2f}%
52-Week High: ₹{ind['week_52_high']:.2f}
52-Week Low: ₹{ind['week_52_low']:.2f}

Technical Indicators:
- RSI (14): {ind['latest_rsi']:.2f}
- MACD: {ind['latest_macd']:.4f} | Signal: {ind['latest_macd_signal']:.4f}
- EMA 20: ₹{ind['latest_ema20']:.2f} | EMA 50: ₹{ind['latest_ema50']:.2f} | EMA 200: ₹{ind['latest_ema200']:.2f}
- ADX: {ind['latest_adx']:.2f} | ATR: {ind['latest_atr']:.2f}
- Bollinger Bands: ₹{ind['latest_bb_lower']:.2f} — ₹{ind['latest_bb_upper']:.2f}
- Support: ₹{ind['support_1']:.2f} | Resistance: ₹{ind['resistance_1']:.2f}
{fund_section}
Technical Score: {score}/100
System Recommendation: {recommendation}

Provide a concise professional analysis:

**TREND SUMMARY** — One-line verdict.
**KEY LEVELS** — Support, resistance, stop-loss.
**INTRADAY VIEW** — Quick trade setup.
**SWING TRADE (1-2 weeks)** — Entry, target, stop-loss.
**POSITIONAL VIEW (1-3 months)** — Medium-term outlook.
{('**FUNDAMENTAL VIEW** — Valuation and financial health assessment.' if fundamentals else '')}
**RISK FACTORS** — Top 3 risks.
**VERDICT** — Clear actionable recommendation with conviction level.

Keep under 350 words. Be direct. Use ₹ for prices."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        timeout=30
    )

    return response.choices[0].message.content


# ==========================
# LOAD STOCK UNIVERSE
# ==========================
stock_universe = load_stock_universe()
all_sectors = sorted(stock_universe["sector"].unique().tolist())


# ==========================
# SESSION STATE
# ==========================
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = "RELIANCE"
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []


# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2rem;">⚡</span>
        <h2 style="margin: 0.3rem 0; font-weight: 800; background: linear-gradient(135deg, #6366f1, #a855f7);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">StockPulse AI</h2>
        <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Professional Stock Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ===== SEARCH (Priority 2) =====
    st.markdown('<p style="color: #64748b; font-size: 0.8rem; font-weight: 600;">🔍 SEARCH COMPANY</p>', unsafe_allow_html=True)
    
    # Build search options list
    search_options = [""] + [f"{row['symbol']} — {row['company_name']}" for _, row in stock_universe.iterrows()]
    
    # Find current index
    current_label = f"{st.session_state.selected_stock} —"
    current_idx = 0
    for i, opt in enumerate(search_options):
        if opt.startswith(current_label):
            current_idx = i
            break
    
    selected = st.selectbox(
        "Search stock", 
        options=search_options, 
        index=current_idx,
        key="stock_search_select",
        placeholder="Type to search... e.g. TCS, Reliance, Banking",
        label_visibility="collapsed"
    )
    
    if selected:
        symbol = selected.split(" — ")[0].strip()
        if symbol != st.session_state.selected_stock:
            st.session_state.selected_stock = symbol
            st.rerun()
    
    st.markdown("---")

    # Sector Browse
    st.markdown('<p style="color: #64748b; font-size: 0.8rem; font-weight: 600;">📂 BROWSE BY SECTOR</p>', unsafe_allow_html=True)
    
    selected_sector = st.selectbox("Sector", ["All"] + all_sectors, key="sector_browse")
    
    if selected_sector != "All":
        sector_stocks = stock_universe[stock_universe["sector"] == selected_sector].head(10)
        cols_per_row = 3
        for i in range(0, len(sector_stocks), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (_, row) in enumerate(sector_stocks.iloc[i:i+cols_per_row].iterrows()):
                if cols[j].button(row["symbol"], key=f"sec_{row['symbol']}", use_container_width=True):
                    st.session_state.selected_stock = row["symbol"]
                    st.rerun()

    st.markdown("---")

    # Settings
    st.markdown('<p style="color: #64748b; font-size: 0.8rem; font-weight: 600;">⚙️ SETTINGS</p>', unsafe_allow_html=True)
    period = st.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    chart_type = st.selectbox("Chart Style", ["Line + Area", "Candlestick"], index=0)

    st.markdown("---")

    # ===== WATCHLIST (Priority 5) =====
    st.markdown('<p style="color: #64748b; font-size: 0.8rem; font-weight: 600;">📋 WATCHLIST</p>', unsafe_allow_html=True)

    wl_input = st.text_input("Add symbol", key="wl_add", placeholder="Add to watchlist...", label_visibility="collapsed")
    if wl_input:
        sym = wl_input.upper().strip()
        if sym and sym not in st.session_state.watchlist:
            st.session_state.watchlist.append(sym)
            st.rerun()

    for ws in st.session_state.watchlist:
        col1, col2 = st.columns([4, 1])
        if col1.button(f"📊 {ws}", key=f"wl_{ws}", use_container_width=True):
            st.session_state.selected_stock = ws
            st.rerun()
        if col2.button("✕", key=f"rm_{ws}"):
            st.session_state.watchlist.remove(ws)
            st.rerun()

    st.markdown("---")

    # ===== PORTFOLIO (Priority 5) =====
    st.markdown('<p style="color: #64748b; font-size: 0.8rem; font-weight: 600;">💼 MY PORTFOLIO</p>', unsafe_allow_html=True)

    with st.expander("Add Holding"):
        pf_symbol = st.text_input("Symbol", key="pf_sym", placeholder="RELIANCE")
        pf_qty = st.number_input("Quantity", min_value=1, value=1, key="pf_qty")
        pf_price = st.number_input("Buy Price (₹)", min_value=0.01, value=100.0, key="pf_price")
        if st.button("Add to Portfolio", key="pf_add", use_container_width=True):
            if pf_symbol:
                st.session_state.portfolio.append({
                    "symbol": pf_symbol.upper().strip(),
                    "qty": pf_qty,
                    "buy_price": pf_price,
                    "date": datetime.now().strftime("%Y-%m-%d")
                })
                st.rerun()

    if st.session_state.portfolio:
        for i, h in enumerate(st.session_state.portfolio):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{h['symbol']}** — {h['qty']} @ ₹{h['buy_price']:.2f}")
            if col2.button("✕", key=f"pf_rm_{i}"):
                st.session_state.portfolio.pop(i)
                st.rerun()
    else:
        st.markdown('<p style="color: #475569; font-size: 0.8rem;">No holdings added yet.</p>', unsafe_allow_html=True)

    # Universe stats
    st.markdown("---")
    st.markdown(f'<p style="color: #475569; font-size: 0.75rem;">📊 {len(stock_universe)} stocks • {len(all_sectors)} sectors</p>', unsafe_allow_html=True)


# ==========================
# MAIN CONTENT
# ==========================

stock = st.session_state.selected_stock

st.markdown("""
<div class="main-header">
    <h1>⚡ StockPulse AI</h1>
    <p>Professional-grade technical analysis powered by AI — NSE stocks</p>
</div>
""", unsafe_allow_html=True)

if stock:
    ticker = stock.upper() + ".NS"
    
    # Get company info from universe
    stock_info = stock_universe[stock_universe["symbol"] == stock.upper()]
    company_name = stock_info.iloc[0]["company_name"] if not stock_info.empty else stock.upper()
    company_sector = stock_info.iloc[0]["sector"] if not stock_info.empty else ""
    company_industry = stock_info.iloc[0]["industry"] if not stock_info.empty else ""

    # Show company name
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <span style="font-size: 1.4rem; font-weight: 700; color: #e2e8f0;">{company_name}</span>
        <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">{stock.upper()}</span>
        <span style="color: #64748b; font-size: 0.85rem; margin-left: 0.5rem;">{company_sector} • {company_industry}</span>
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner(f"Loading {stock.upper()} data..."):
            data = get_stock_data(ticker, period)

        if data is None or data.empty:
            st.error(f"Could not find data for {stock.upper()}. Check the symbol and try again.")
            st.stop()

        ind = compute_indicators(data)
        score, recommendation, rec_class, reasons = compute_score(ind)
        
        # Fundamentals loaded lazily — only when needed (tabs, AI, report)
        @st.cache_data(ttl=600)
        def _get_fund(t):
            return get_fundamentals(t)
        
        fund_info = None  # loaded on demand below

        # ============ TOP METRICS (3+3 grid for mobile) ============
        delta_color = "metric-delta-up" if ind["price_change"] >= 0 else "metric-delta-down"
        delta_arrow = "▲" if ind["price_change"] >= 0 else "▼"

        row1_c1, row1_c2, row1_c3 = st.columns(3)

        with row1_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">₹{ind['current_price']:.2f}</div>
                <div class="{delta_color}">{delta_arrow} {abs(ind['price_change']):.2f} ({ind['price_change_pct']:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with row1_c2:
            rsi_color = "#22c55e" if ind["latest_rsi"] > 50 else "#ef4444" if ind["latest_rsi"] < 40 else "#eab308"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RSI (14)</div>
                <div class="metric-value" style="color: {rsi_color}">{ind['latest_rsi']:.1f}</div>
                <div style="color: #64748b; font-size: 0.75rem;">{"Overbought" if ind['latest_rsi'] > 70 else "Oversold" if ind['latest_rsi'] < 30 else "Neutral"}</div>
            </div>
            """, unsafe_allow_html=True)

        with row1_c3:
            macd_color = "#22c55e" if ind["latest_macd"] > ind["latest_macd_signal"] else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MACD</div>
                <div class="metric-value" style="color: {macd_color}">{ind['latest_macd']:.2f}</div>
                <div style="color: #64748b; font-size: 0.75rem;">Signal: {ind['latest_macd_signal']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        row2_c1, row2_c2, row2_c3 = st.columns(3)

        with row2_c1:
            adx_label = "Strong" if ind["latest_adx"] > 25 else "Weak"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ADX (Trend)</div>
                <div class="metric-value">{ind['latest_adx']:.1f}</div>
                <div style="color: #64748b; font-size: 0.75rem;">{adx_label} Trend</div>
            </div>
            """, unsafe_allow_html=True)

        with row2_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">52W Range</div>
                <div class="metric-value" style="font-size: 1.1rem;">₹{ind['week_52_low']:.0f} — ₹{ind['week_52_high']:.0f}</div>
                <div style="color: #64748b; font-size: 0.75rem;">ATR: {ind['latest_atr']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with row2_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Technical Score</div>
                <div class="metric-value" style="font-size: 1.8rem;">{score}<span style="font-size: 0.9rem; color: #64748b">/100</span></div>
                <div class="score-badge {rec_class}">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

        # ============ MAIN LAYOUT ============
        chart_col, analysis_col = st.columns([2.5, 1])

        with chart_col:
            st.markdown('<div class="section-header">📈 Technical Chart</div>', unsafe_allow_html=True)
            fig = create_main_chart(ind, stock.upper())

            if chart_type == "Candlestick":
                clean_data = data.dropna()
                ohlc_data = clean_data.copy()
                for col in ["Open", "High", "Low", "Close"]:
                    ohlc_data[col] = flatten_col(ohlc_data[col])
                fig.data = [t for t in fig.data if t.name != "Price"]
                fig.add_trace(go.Candlestick(
                    x=ohlc_data.index, open=ohlc_data["Open"], high=ohlc_data["High"],
                    low=ohlc_data["Low"], close=ohlc_data["Close"], name="OHLC",
                    increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                ), row=1, col=1)
                fig.update_layout(xaxis_rangeslider_visible=False)

            st.plotly_chart(fig, use_container_width=True)

        with analysis_col:
            # ===== AI RESEARCH REPORT (Priority 4) =====
            st.markdown('<div class="section-header">🔬 Research Report</div>', unsafe_allow_html=True)

            # Lazy load fundamentals for report
            if fund_info is None:
                fund_info = _get_fund(ticker)
            report = build_ai_report(ind, score, recommendation, reasons, fund_info)

            # Trend
            trend_text, trend_color = report["trend"]
            st.markdown(f"""
            <div class="ai-report" style="padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">TREND</span>
                    <span style="color: {trend_color}; font-weight: 700; font-size: 1.1rem;">{trend_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Technical Signals
            st.markdown('<p style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-top: 0.5rem;">TECHNICAL SIGNALS</p>', unsafe_allow_html=True)
            for signal_text, is_positive in report["tech_signals"]:
                icon = "✓" if is_positive else "✗"
                color = "#22c55e" if is_positive else "#ef4444"
                st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 0.25rem 0; gap: 0.5rem;">
                    <span style="color: {color}; font-weight: 700;">{icon}</span>
                    <span style="color: #cbd5e1; font-size: 0.85rem;">{signal_text}</span>
                </div>
                """, unsafe_allow_html=True)

            # Fundamental Signals
            if report["fund_signals"]:
                st.markdown('<p style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-top: 0.8rem;">FUNDAMENTAL STRENGTH</p>', unsafe_allow_html=True)
                for signal_text, is_positive in report["fund_signals"]:
                    icon = "✓" if is_positive else "✗"
                    color = "#22c55e" if is_positive else "#ef4444"
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; padding: 0.25rem 0; gap: 0.5rem;">
                        <span style="color: {color}; font-weight: 700;">{icon}</span>
                        <span style="color: #cbd5e1; font-size: 0.85rem;">{signal_text}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Risk Level
            risk_text, risk_class = report["risk"]
            st.markdown(f"""
            <div style="margin-top: 0.8rem; display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">RISK LEVEL</span>
                <span class="risk-badge {risk_class}">{risk_text}</span>
            </div>
            """, unsafe_allow_html=True)

            # Strategy
            st.markdown(f"""
            <div style="margin-top: 0.8rem; background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.8rem;">
                <p style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; margin: 0 0 0.3rem 0;">SUGGESTED STRATEGY</p>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0; line-height: 1.5;">{report['strategy']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Key Levels
            st.markdown('<div class="section-header">📍 Key Levels</div>', unsafe_allow_html=True)
            levels = [
                ("Resistance 2", ind["resistance_2"], "#ef4444"),
                ("Resistance 1", ind["resistance_1"], "#f87171"),
                ("Pivot", ind["pivot"], "#6366f1"),
                ("Support 1", ind["support_1"], "#4ade80"),
                ("Support 2", ind["support_2"], "#22c55e"),
            ]
            for name, val, color in levels:
                st.markdown(f"""
                <div class="indicator-card" style="padding: 0.6rem 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span class="indicator-name">{name}</span>
                        <span class="indicator-value" style="color: {color}; font-size: 0.95rem;">₹{val:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ============ AI DEEP ANALYSIS ============
        st.markdown('<div class="section-header">🧠 AI-Powered Deep Analysis</div>', unsafe_allow_html=True)

        ai_col1, ai_col2 = st.columns([1, 4])
        with ai_col1:
            run_ai = st.button("⚡ Generate AI Analysis", use_container_width=True)

        if run_ai:
            try:
                if fund_info is None:
                    fund_info = _get_fund(ticker)
                fund_data = None
                if fund_info:
                    mcap = fund_info.get("marketCap", 0)
                    fund_data = {
                        "mcap": f"₹{mcap/1e7:.0f} Cr" if mcap else "N/A",
                        "pe": f"{fund_info['trailingPE']:.2f}" if fund_info.get("trailingPE") else "N/A",
                        "eps": f"₹{fund_info['trailingEps']:.2f}" if fund_info.get("trailingEps") else "N/A",
                        "roe": f"{fund_info['returnOnEquity']*100:.2f}%" if fund_info.get("returnOnEquity") else "N/A",
                        "de": f"{fund_info['debtToEquity']:.2f}" if fund_info.get("debtToEquity") else "N/A",
                        "rev_growth": f"{fund_info['revenueGrowth']*100:.2f}%" if fund_info.get("revenueGrowth") else "N/A",
                        "margin": f"{fund_info['profitMargins']*100:.2f}%" if fund_info.get("profitMargins") else "N/A",
                        "div_yield": f"{fund_info['dividendYield']*100:.2f}%" if fund_info.get("dividendYield") else "N/A",
                    }
                with st.spinner("Generating AI analysis..."):
                    ai_text = get_ai_analysis(stock.upper(), ind, score, recommendation, fund_data)
                    st.session_state["ai_text"] = ai_text
                    st.session_state["ai_stock"] = stock.upper()
            except Exception as ai_err:
                st.error(f"AI Analysis unavailable: {ai_err}")

        if "ai_text" in st.session_state and st.session_state.get("ai_stock") == stock.upper():
            st.markdown(f'<div class="ai-analysis-box">{st.session_state["ai_text"]}</div>', unsafe_allow_html=True)
        elif not run_ai:
            st.markdown("""
            <div class="ai-analysis-box" style="text-align: center; padding: 2rem; color: #64748b;">
                Click <strong>Generate AI Analysis</strong> for a detailed AI-powered breakdown.<br>
                <span style="font-size: 0.8rem;">Uses OpenAI API — one call per click.</span>
            </div>
            """, unsafe_allow_html=True)

        # ============ TABS ============
        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Recent Data", "📊 Compare", "📐 Indicators", "🏢 Fundamentals", "💼 Portfolio"])

        with tab1:
            recent_raw = data.tail(15).copy()
            # Flatten MultiIndex columns
            if hasattr(recent_raw.columns, 'levels'):
                recent_raw.columns = [c[0] if isinstance(c, tuple) else c for c in recent_raw.columns]
            recent_display = pd.DataFrame(index=recent_raw.index)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in recent_raw.columns:
                    recent_display[col] = pd.to_numeric(recent_raw[col], errors="coerce")
            recent_display["RSI"] = ind["rsi"].reindex(recent_display.index)
            recent_display["EMA20"] = ind["ema20"].reindex(recent_display.index)
            st.dataframe(recent_display.style.format("{:.2f}", na_rep="—"), use_container_width=True, height=400)

        with tab2:
            st.markdown("Compare up to 4 stocks side by side.")
            compare_stocks = st.text_input("Enter symbols separated by comma", value="RELIANCE, TCS, INFY", key="compare_input")
            if compare_stocks:
                symbols = [s.strip().upper() for s in compare_stocks.split(",")][:4]
                compare_fig = go.Figure()
                for sym in symbols:
                    try:
                        comp_data = yf.download(sym + ".NS", period=period, auto_adjust=True, progress=False)
                        if not comp_data.empty:
                            comp_close = flatten_col(comp_data["Close"])
                            normalized = (comp_close / comp_close.iloc[0] - 1) * 100
                            compare_fig.add_trace(go.Scatter(x=normalized.index, y=normalized, name=sym, mode="lines", line=dict(width=2)))
                    except Exception:
                        pass
                compare_fig.update_layout(
                    height=450, template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0f1629",
                    font=dict(family="Inter", color="#94a3b8"), yaxis_title="Return (%)",
                    margin=dict(l=60, r=20, t=30, b=30), hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                compare_fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
                compare_fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)")
                st.plotly_chart(compare_fig, use_container_width=True)

        with tab3:
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Momentum**")
                st.markdown(f"| Indicator | Value |\n|---|---|\n| RSI (14) | {ind['latest_rsi']:.2f} |\n| Stochastic %K | {ind['latest_stoch_k']:.2f} |\n| MACD | {ind['latest_macd']:.4f} |\n| MACD Signal | {ind['latest_macd_signal']:.4f} |")
            with d2:
                st.markdown("**Trend**")
                st.markdown(f"| Indicator | Value |\n|---|---|\n| EMA 9 | ₹{ind['latest_ema9']:.2f} |\n| EMA 20 | ₹{ind['latest_ema20']:.2f} |\n| EMA 50 | ₹{ind['latest_ema50']:.2f} |\n| EMA 200 | ₹{ind['latest_ema200']:.2f} |\n| ADX | {ind['latest_adx']:.2f} |")
            with d3:
                st.markdown("**Volatility**")
                st.markdown(f"| Indicator | Value |\n|---|---|\n| BB Upper | ₹{ind['latest_bb_upper']:.2f} |\n| BB Lower | ₹{ind['latest_bb_lower']:.2f} |\n| ATR (14) | {ind['latest_atr']:.2f} |\n| 52W High | ₹{ind['week_52_high']:.2f} |\n| 52W Low | ₹{ind['week_52_low']:.2f} |")

        with tab4:
            if fund_info is None:
                with st.spinner("Loading fundamentals..."):
                    fund_info = _get_fund(ticker)
            if fund_info:
                st.markdown(f"**{company_name}** — Fundamental Analysis")
                f1, f2, f3 = st.columns(3)

                with f1:
                    st.markdown("**Valuation**")
                    pe = fund_info.get("trailingPE", "N/A")
                    fwd_pe = fund_info.get("forwardPE", "N/A")
                    pb = fund_info.get("priceToBook", "N/A")
                    mcap = fund_info.get("marketCap", 0)
                    mcap_str = f"₹{mcap/1e7:.0f} Cr" if mcap else "N/A"
                    ev_ebitda = fund_info.get("enterpriseToEbitda", "N/A")
                    st.markdown(f"| Metric | Value |\n|---|---|\n| Market Cap | {mcap_str} |\n| P/E (TTM) | {pe if pe == 'N/A' else f'{pe:.2f}'} |\n| Forward P/E | {fwd_pe if fwd_pe == 'N/A' else f'{fwd_pe:.2f}'} |\n| P/B Ratio | {pb if pb == 'N/A' else f'{pb:.2f}'} |\n| EV/EBITDA | {ev_ebitda if ev_ebitda == 'N/A' else f'{ev_ebitda:.2f}'} |")

                with f2:
                    st.markdown("**Profitability**")
                    roe = fund_info.get("returnOnEquity", "N/A")
                    roa = fund_info.get("returnOnAssets", "N/A")
                    margin = fund_info.get("profitMargins", "N/A")
                    ops_margin = fund_info.get("operatingMargins", "N/A")
                    eps = fund_info.get("trailingEps", "N/A")
                    st.markdown(f"| Metric | Value |\n|---|---|\n| EPS | {eps if eps == 'N/A' else f'₹{eps:.2f}'} |\n| ROE | {roe if roe == 'N/A' else f'{roe*100:.2f}%'} |\n| ROA | {roa if roa == 'N/A' else f'{roa*100:.2f}%'} |\n| Profit Margin | {margin if margin == 'N/A' else f'{margin*100:.2f}%'} |\n| Operating Margin | {ops_margin if ops_margin == 'N/A' else f'{ops_margin*100:.2f}%'} |")

                with f3:
                    st.markdown("**Financial Health**")
                    de = fund_info.get("debtToEquity", "N/A")
                    cr = fund_info.get("currentRatio", "N/A")
                    rev = fund_info.get("totalRevenue", 0)
                    rev_str = f"₹{rev/1e7:.0f} Cr" if rev else "N/A"
                    rev_growth = fund_info.get("revenueGrowth", "N/A")
                    div_yield = fund_info.get("dividendYield", "N/A")
                    st.markdown(f"| Metric | Value |\n|---|---|\n| Revenue | {rev_str} |\n| Revenue Growth | {rev_growth if rev_growth == 'N/A' else f'{rev_growth*100:.2f}%'} |\n| Debt/Equity | {de if de == 'N/A' else f'{de:.2f}'} |\n| Current Ratio | {cr if cr == 'N/A' else f'{cr:.2f}'} |\n| Dividend Yield | {div_yield if div_yield == 'N/A' else f'{div_yield*100:.2f}%'} |")
            else:
                st.warning("Fundamental data unavailable for this stock.")

        with tab5:
            st.markdown("**Portfolio Tracker**")
            if st.session_state.portfolio:
                pf_data = []
                total_invested = 0
                total_current = 0

                for h in st.session_state.portfolio:
                    try:
                        pf_ticker_data = yf.download(h["symbol"] + ".NS", period="5d", auto_adjust=True, progress=False)
                        if not pf_ticker_data.empty:
                            curr = float(flatten_col(pf_ticker_data["Close"]).iloc[-1])
                        else:
                            curr = h["buy_price"]
                    except Exception:
                        curr = h["buy_price"]

                    invested = h["qty"] * h["buy_price"]
                    current = h["qty"] * curr
                    pnl = current - invested
                    pnl_pct = (pnl / invested) * 100

                    total_invested += invested
                    total_current += current

                    pf_data.append({
                        "Symbol": h["symbol"],
                        "Qty": h["qty"],
                        "Buy Price": f"₹{h['buy_price']:.2f}",
                        "Current": f"₹{curr:.2f}",
                        "Invested": f"₹{invested:.2f}",
                        "Current Value": f"₹{current:.2f}",
                        "P&L": f"₹{pnl:.2f}",
                        "P&L %": f"{pnl_pct:.2f}%"
                    })

                pf_df = pd.DataFrame(pf_data)
                st.dataframe(pf_df, use_container_width=True, hide_index=True)

                total_pnl = total_current - total_invested
                total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
                pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"

                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Total Invested", f"₹{total_invested:,.2f}")
                tc2.metric("Current Value", f"₹{total_current:,.2f}")
                tc3.markdown(f'<div class="metric-card"><div class="metric-label">Total P&L</div><div class="metric-value" style="color: {pnl_color}">₹{total_pnl:,.2f}</div></div>', unsafe_allow_html=True)
                tc4.markdown(f'<div class="metric-card"><div class="metric-label">Returns</div><div class="metric-value" style="color: {pnl_color}">{total_pnl_pct:.2f}%</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #475569;">Add holdings from the sidebar to track your portfolio here.</p>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.markdown(f"<p style='color: #64748b; font-size: 0.85rem;'>Debug: {type(e).__name__}: {str(e)}</p>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 2rem;">
    <p style="color: #475569; font-size: 0.8rem;">
        ⚡ StockPulse AI — For educational purposes only. Not financial advice.<br>
        Data from Yahoo Finance. AI analysis via OpenAI.
    </p>
</div>
""", unsafe_allow_html=True)
