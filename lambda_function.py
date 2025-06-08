import psycopg2
import json
import re
import requests
import pandas as pd
from io import BytesIO
import boto3
import uuid
from urllib.parse import urlparse
import os


def extract_metadata_from_url(url):
    pattern = re.compile(
        r"clientProject/(?P<project_id>\d+)/user_id/(?P<user_id>\d+)/"
        r"planning_scenario/(?P<scenario_id>\d+)/table_name/(?P<table_name>[^/]+)/"
    )
    match = pattern.search(url)
    if match:
        return match.groupdict()
    else:
        raise ValueError("URL format invalid or missing required parts.")


def upload_cleaned_file_to_s3(df, original_file_url, aws_config):
    try:
        metadata = extract_metadata_from_url(original_file_url)
    except ValueError as e:
        print(f"❌ Failed to extract metadata: {e}")
        return

    # Use filename from URL
    original_filename = urlparse(original_file_url).path.split("/")[-1].split("?")[0]
    file_base_name = original_filename.split(".")[0] if original_filename else metadata["table_name"]
    unique_file_name = f"{uuid.uuid4()}_{file_base_name}.csv"

    s3_key = (
        f"fpa/cleanedfiles/clientProject/{metadata['project_id']}/user_id/{metadata['user_id']}/"
        f"planning_scenario/{metadata['scenario_id']}/table_name/{metadata['table_name']}/{unique_file_name}"
    )

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_config["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_config["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_config["AWS_REGION"],
    )

    file_buffer = BytesIO()
    df.to_csv(file_buffer, index=False)
    file_buffer.seek(0)

    s3_client.upload_fileobj(file_buffer, aws_config["AWS_S3_BUCKET_DOCUMENT"], s3_key)
    presigned_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": aws_config["AWS_S3_BUCKET_DOCUMENT"], "Key": s3_key},
        ExpiresIn=43200,
    )

    print(f"✅ Uploaded cleaned file to: {presigned_url}\n")
    return presigned_url

def load_single_file(file_url, sample_rows=5):
    response = requests.get(file_url)
    response.raise_for_status()
    file_bytes = BytesIO(response.content)

    path = urlparse(file_url).path

    if path.lower().endswith(".csv"):
        df = pd.read_csv(file_bytes)
        sheet_name = path.split("/")[-1].split(".csv")[0]
    else:
        xl = pd.ExcelFile(file_bytes)
        sheet_name = xl.sheet_names[0]
        df = xl.parse(sheet_name)

    # Fill nulls for preview only
    columns = df.columns.tolist()
    samples = df.head(sample_rows).to_dict(orient="records")

    return sheet_name, df, {
        "columns": columns,
        "samples": samples
    }


def call_llm(prompt, api_url, token=None):
    payload = {
        "model": "Qwen/Qwen3-14B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    response_text = response.json()['choices'][0]['message']['content']
    # Remove <think>...</think> tags if any
    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()


def match_columns(sheet_columns, table_name, table_columns, api_url, token=None):
    prompt = f"""
        Match the Excel sheet columns:
        {json.dumps(sheet_columns)}

        To the database table '{table_name}' with columns:
        {json.dumps(table_columns)}

        Return a JSON mapping in the format:
        {{
        "ExcelColumn1": "DBColumn1",
        "ExcelColumn2": "DBColumn2"
        }}
        """
    return call_llm(prompt, api_url, token)


def fetch_db_metadata(host, dbname, user, password, port):
    conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name LIKE 'fpa_%'
        ORDER BY table_name, ordinal_position;
    """)
    base_columns = cur.fetchall()

    table_columns_map = {}
    for table_name, column_name, data_type in base_columns:
        table_columns_map.setdefault(table_name, []).append({"column_name": column_name, "data_type": data_type})

    generic_fields = [col for col in table_columns_map.get('fpa_genericdimension', []) if col["column_name"] != "id"]

    for table_name in list(table_columns_map.keys()):
        if table_name == 'fpa_genericdimension':
            continue

        columns = table_columns_map[table_name]
        col_names = [col["column_name"] for col in columns]

        if "genericdimension_ptr_id" in col_names:
            inherited_fields = [{"column_name": "id", "data_type": "character varying"}] + generic_fields
            existing_cols = {col["column_name"] for col in columns}
            for gf in reversed(inherited_fields):
                if gf["column_name"] not in existing_cols:
                    columns.insert(0, gf)

            table_columns_map[table_name] = [col for col in columns if col["column_name"] != "genericdimension_ptr_id"]

    table_columns_map.pop('fpa_genericdimension', None)

    cur.close()
    conn.close()

    return table_columns_map


def match_sheet_to_table(sheet_name, sheet_columns, db_tables, api_url, token=None):
    table_column_map = {table: [col['column_name'] for col in cols] for table, cols in db_tables.items()}

    prompt = f"""
        You are an expert data analyst.

        Your task is to match the following Excel sheet to the most appropriate database table.

        ### Sheet Details:
        - Sheet name: "{sheet_name}"
        - Sheet columns: {json.dumps(sheet_columns, indent=2)}

        ### Available DB Tables:
        {json.dumps(table_column_map, indent=2)}

        ### Response Format (strict JSON):
        {{
        "matched_table": "best_matched_table_name",
        "confidence": 85,
        "reason": "why this match was selected"
        }}
        """
    return call_llm(prompt, api_url, token)


def fill_nulls(df, sheet_name):
    print(f"\n🧼 Null Replacement Log for Sheet: {sheet_name}")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
                print(f"✅ Replaced {null_count} nulls with 0 in numeric column: '{col}'")
            else:
                df[col] = df[col].fillna('nan')
                print(f"✅ Replaced {null_count} nulls with 'nan' in non-numeric column: '{col}'")
        else:
            print(f"👌 No nulls found in column: '{col}'")


def lambda_handler(event, context):
    """
    Expected event format is the payload dict you previously used.
    If your Lambda is behind API Gateway, parse event['body'] as JSON.
    """
    try:
        # If event is from API Gateway proxy, parse the body
        if "body" in event and isinstance(event["body"], str):
            payload = json.loads(event["body"])
        else:
            payload = event

        # You can also optionally get API_URL from environment variables
        api_url = os.environ.get("API_URL")
        if not api_url:
            raise ValueError("API_URL environment variable not set")

        planning_scenario_id = str(payload["planning_scenario_id"])
        project_id = str(payload["project_id"])
        user_id = str(payload["user_id"])

        aws_config = {
            "AWS_ACCESS_KEY_ID": payload["aws_access_key_id"],
            "AWS_SECRET_ACCESS_KEY": payload["aws_secret_access_key"],
            "AWS_REGION": payload["aws_region"],
            "AWS_S3_BUCKET_DOCUMENT": "dev-ai-analytics-private"  # or pass from payload if dynamic
        }

        # DB config - either env vars or hardcoded
        DB_CONFIG = {
            "host": os.environ.get("DB_HOST", "ai-analytics.cws5wr16kwar.ap-south-1.rds.amazonaws.com"),
            "dbname": os.environ.get("DB_NAME", "dev-analytics"),
            "user": os.environ.get("DB_USER", "analytics_admin"),
            "password": os.environ.get("DB_PASSWORD", "sfsNOB5U2y2SvJ8EAZDx"),
            "port": int(os.environ.get("DB_PORT", "5432"))
        }

        file_urls = [file_info["presigned_url"] for file_info in payload["files"]]

        db_tables = fetch_db_metadata(**DB_CONFIG)
        print("\n📘 Loaded DB Tables:")
        print(json.dumps({t: [col["column_name"] for col in cols] for t, cols in db_tables.items()}, indent=2))

        matched_tables = set()
        upload_results = []

        for file_url in file_urls:
            print(f"\n📄 Processing file: {file_url}")
            try:
                sheet_name, df, preview = load_single_file(file_url)
            except Exception as e:
                error_msg = f"Failed to load file {file_url}: {e}"
                print("❌", error_msg)
                upload_results.append({"file_url": file_url, "error": error_msg})
                continue

            sheet_columns = preview["columns"]

            # Available tables excluding matched ones
            available_tables = {t: cols for t, cols in db_tables.items() if t not in matched_tables}
            if not available_tables:
                warning_msg = "No available DB tables left to match."
                print("⚠️", warning_msg)
                upload_results.append({"file_url": file_url, "warning": warning_msg})
                break

            match_result = match_sheet_to_table(sheet_name, sheet_columns, available_tables, api_url, payload.get("token"))
            print("🧠 LLM Table Match:\n", match_result)

            try:
                    matched_table = json.loads(match_result)["matched_table"]
                    matched_tables.add(matched_table)

                    db_columns = [col["column_name"] for col in db_tables[matched_table]]
                    col_map_response = match_columns(sheet_columns, matched_table, db_columns, api_url, payload.get("token"))
                    print("🔗 Column Mapping:\n", col_map_response)

                    col_map_json = json.loads(col_map_response)
                    df.rename(columns=col_map_json, inplace=True)

                    fill_nulls(df, sheet_name)

                    original_filename = file_url.split("/")[-1].split("?")[0]

                    presigned_url =upload_cleaned_file_to_s3(
                    df=df,
                    original_file_url=file_url,
                    aws_config=aws_config
                    )

                    upload_results.append({"file_url": file_url, "uploaded_url": presigned_url})

            except Exception as e:
                    error_msg = f"Error processing sheet '{sheet_name}': {e}"
                    print("❌", error_msg)
                    upload_results.append({"file_url": file_url, "error": error_msg})

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Processing complete",
                "results": upload_results
            }),
            "headers": {"Content-Type": "application/json"}
            }

    except Exception as e:
            print("❌ Fatal error:", e)
            return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
