import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv("secrets.env")

# Config
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"
TARGET_PER_TYPE = 80
SLEEP_BETWEEN_CALLS = 2.5
INPUT_FILE = "shopping_tools/shopping_tool_descriptions.json"
OUTPUT_FILE = "shopping_tools/shopping_tool_perturbations_addl.json"

PERTURBATION_TYPES = [
    # "description_mismatch",
    # "paraphrase",
    # "vague_description",
    # "description_overload",
    # "noise_injection",
    "negation_injection",
    "prefix_noise_injection",
    "interleaved_noise_injection",
]

# Load tools
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file {INPUT_FILE} not found.")

with open(INPUT_FILE, "r") as f:
    SHOPPING_TOOLS = json.load(f)

# Groq Client
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment.")

client = Groq(api_key=API_KEY)

# Prompt template
PROMPT_TEMPLATE = """
You are helping generate controlled perturbations of tool descriptions
for an AI agent interpretability experiment.

Tools and Original Descriptions:
{tools_json}

Perturbation Type:
{perturbation_type}

Definition:
{perturbation_definition}

- Generate BETWEEN 10 AND 12 variants.
- Each variant MUST contain descriptions for ALL tools.
- Return JSON only.
- Keep tool names exactly the same as the keys in the input JSON. 
- Maintain grammatical English. 
- Avoid repeating earlier phrasing patterns.

Format:

{{
 "{perturbation_type}": [
   {{
     "ToolA": "...",
     "ToolB": "...",
     "ToolC": "...",
     "ToolD": "...",
     "ToolE": "..."
   }}
 ]
}}
"""

# "description_mismatch": "Swap descriptions between tools while keeping tool names fixed.",
# "paraphrase": "Rewrite descriptions using different wording but same meaning.",
# "suffix_noise_injection": "Add irrelevant or distracting sentences but keep core meaning intact.",
# "vague_description": "Rewrite descriptions to be more generic and less specific.",
# "description_overload": "Make descriptions longer and more verbose with extra details.",
PERTURBATION_DEFINITIONS = {
    "negation_injection": "Rewrite the descriptions by introducing negation or double-negation expressions while preserving the original meaning. The tool capability must remain the same. Do NOT negate the functionality of the tool.",
    "prefix_noise_injection": "Add irrelevant or distracting sentences at the beginning of the sentence but keep core meaning intact.",
    "interleaved_noise_injection": "Add irrelevant or distracting sentences interleaved between the existing sentences but keep core meaning intact.",
}


def clean_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


# Initialize dataset
dataset = {t: [] for t in PERTURBATION_TYPES}
seen = {t: set() for t in PERTURBATION_TYPES}


def add_variants(data, perturbation_type):
    if perturbation_type not in data:
        return

    for variant in data[perturbation_type]:
        if set(variant.keys()) != set(SHOPPING_TOOLS.keys()):
            print("Rejected variant (missing tools)")
            continue

        key = json.dumps(variant, sort_keys=True)

        if key not in seen[perturbation_type]:
            seen[perturbation_type].add(key)
            dataset[perturbation_type].append(variant)


tools_json_str = json.dumps(SHOPPING_TOOLS, indent=2)

iteration = 1

for perturbation_type in PERTURBATION_TYPES:
    print(f"\nGenerating perturbations for: {perturbation_type}")

    while len(dataset[perturbation_type]) < TARGET_PER_TYPE:
        print(f"\nBatch {iteration} | {perturbation_type}")

        try:
            prompt = PROMPT_TEMPLATE.format(
                tools_json=tools_json_str,
                perturbation_type=perturbation_type,
                perturbation_definition=PERTURBATION_DEFINITIONS[perturbation_type],
            )

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                top_p=0.95,
                response_format={"type": "json_object"},
                max_tokens=4096,
            )

            raw_text = response.choices[0].message.content

            if not raw_text:
                raise ValueError("Empty response")

            cleaned = clean_json(raw_text)

            # print("\nRAW OUTPUT SAMPLE:\n", cleaned[:800])

            data = json.loads(cleaned)

            add_variants(data, perturbation_type)

            print(f"{perturbation_type}: {len(dataset[perturbation_type])}")

            iteration += 1
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print("Error:", e)
            time.sleep(5)


final_dataset = {t: dataset[t][:TARGET_PER_TYPE] for t in PERTURBATION_TYPES}

with open(OUTPUT_FILE, "w") as f:
    json.dump(final_dataset, f, indent=2)

print(f"\nDataset saved to {OUTPUT_FILE}")
