# Author: Dayeon

import openai
import os
import json
import time
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("EssaySampleText.json", "r", encoding="utf-8") as f:
    essays = json.load(f)

if not os.path.exists("summaries"):
    os.makedirs("summaries")

def clean_filename(text):
    """Remove invalid characters for filenames"""
    return re.sub(r'[\\/*?:"<>|]', "", text)[:50]

def generate_summary(text):
    """Summarize essay using OpenAI Chat API"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize the following essay in a concise and informative way."},
                {"role": "user", "content": text[:4000]}  # Truncate to fit token limit
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Loop through essays
for i, essay in enumerate(essays):
    title = essay.get("Title", "").strip()
    text = essay.get("Text", "").strip()

    if not text or len(text.split()) < 50:
        print(f"[{i}] Skipped empty or too-short essay: {title or 'Untitled'}")
        continue

    print(f"[{i}] Summarizing: {title}")

    summary = generate_summary(text)
    safe_title = clean_filename(title or f"essay_{i+1}")
    filename = os.path.join("summaries", f"{i+1}_{safe_title}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n\nSummary:\n{summary}")

    print(f"Saved: {filename}")
    time.sleep(1.5)  

print("All done.")