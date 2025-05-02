# main.py (Portfolio version)

import os
import pandas as pd
from flask import Flask, jsonify
from datetime import datetime, timedelta
from supabase import create_client
import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_SHEET_MAP = os.getenv("TABLE_SHEET_MAP")  # Example: "table1:Sheet1,table2:Sheet2"
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
DAY_WINDOW = int(os.getenv("DAY_WINDOW", 7))

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Google Sheets credentials (from env-specified file path)
creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")  # masked
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
gc = gspread.authorize(creds)
sh = gc.open_by_key(SPREADSHEET_ID)

# Utility: paginated data fetch
def fetch_all_rows(table, cutoff_iso):
    all_data = []
    limit = 1000
    offset = 0

    while True:
        query = (
            supabase
            .table(table)
            .select("*")
            .gte("timestamp", cutoff_iso)
            .range(offset, offset + limit - 1)
        )
        response = query.execute()
        data = response.data or []

        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit

    return all_data

@app.route("/update", methods=["POST", "GET"])
def update_all():
    if not TABLE_SHEET_MAP:
        return jsonify({"status": "error", "message": "Mapping not defined"}), 400

    mapping = dict(pair.split(":") for pair in TABLE_SHEET_MAP.split(","))
    results = {}
    cutoff_iso = (datetime.utcnow() - timedelta(days=DAY_WINDOW)).isoformat()

    for table, sheet_name in mapping.items():
        try:
            data = fetch_all_rows(table, cutoff_iso)
            if not data:
                results[table] = "no data"
                continue

            df = pd.DataFrame(data).fillna(0)
            worksheet = sh.worksheet(sheet_name)
            worksheet.clear()
            worksheet.update([df.columns.tolist()] + df.values.tolist())

            results[table] = f"{len(df)} rows written"
        except Exception as e:
            results[table] = f"error: {str(e)}"

    return jsonify({"status": "complete", "results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
