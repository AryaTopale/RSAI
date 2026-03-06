import csv
import os
import time

from dotenv import load_dotenv

# from google.genai import Client
from groq import Groq
from tqdm import tqdm

load_dotenv("secrets.env")

# -------------------------
# Config
# -------------------------

# API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY = os.getenv("GROQ_API_KEY")

TOTAL_SAMPLES = 2000
BATCH_SIZE = 25
SLEEP_BETWEEN_CALLS = 5

OUTPUT_FILE = "data/finance_positives.csv"

TOOLS = {
    "GetStockPrice": "Retrieves the current real-time trading price for a specified stock ticker symbol.",
    "ExecuteTrade": "Submits a buy or sell order for a specified number of shares of a given stock.",
    "GetHistoricalData": "Fetches past price and volume data for a stock over a defined historical date range.",
    "AnalyzeSentiment": "Aggregates and scores recent financial news articles to determine the overall market sentiment for a company.",
    "CalculatePortfolioValue": "Computes the total current financial value of all assets and equities currently held in the user's account.",
}

# -------------------------
# Gemini Client
# -------------------------

# client = Client(api_key=API_KEY)

# -------------------------
# Groq Client
# -------------------------

client = Groq(api_key=API_KEY)

# -------------------------
# Prompt builder
# -------------------------


def build_prompt(batch_size):

    tool_desc = "\n".join([f"{k}: {v}" for k, v in TOOLS.items()])

    return f"""
You are generating training data for an LLM tool-calling dataset.

Tools available:
{tool_desc}

Rules:
- Generate {batch_size} examples
- Format: Query,Tool
- Only ONE tool per query
- Each query must clearly correspond to one tool
- Use only provided tool names
- No numbering
- No explanations
- CSV format only

Example:
Query,Tool
What is the price of Tesla stock?,GetStockPrice
Buy 10 shares of Apple.,ExecuteTrade

Generate {batch_size} examples.
"""


# -------------------------
# Generate Dataset
# -------------------------

with open(OUTPUT_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    # writer.writerow(["Query", "Tool"])

    batches = TOTAL_SAMPLES // BATCH_SIZE
    generated = 0

    for _ in tqdm(range(batches)):
        prompt = build_prompt(BATCH_SIZE)

        # response = client.models.generate_content(
        #     model="gemini-2.5-flash", contents=prompt
        # )

        # text = response.text.strip()

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        text = (response.choices[0].message.content or "").strip()

        for line in text.split("\n"):
            if "," not in line:
                continue

            query, tool = line.split(",", 1)

            if tool.strip() not in TOOLS:
                continue

            writer.writerow([query.strip(), tool.strip()])
            generated += 1

        time.sleep(SLEEP_BETWEEN_CALLS)

print(f"Generated {generated} samples")
