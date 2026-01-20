import json

import boto3

lambda_client = boto3.client("lambda", region_name="us-east-2")
s3 = boto3.client("s3")

BUCKET = "automata-learning-will-2026"
PREFIX = "benchmarks/"
NUM_TRIALS = 3

paginator = s3.get_paginator("list_objects_v2")
tlsf_files = []
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        if obj["Key"].endswith(".tlsf"):
            tlsf_files.append(obj["Key"])

print(f"Found {len(tlsf_files)} TLSF files")
print(f"Invoking {len(tlsf_files) * NUM_TRIALS} Lambda functions...")

for tlsf_key in tlsf_files:
    for trial in range(1, NUM_TRIALS + 1):
        payload = {"bucket": BUCKET, "tlsf_key": tlsf_key, "trial": trial}

        lambda_client.invoke(
            FunctionName="automata-learning",
            InvocationType="Event",
            Payload=json.dumps(payload),
        )
        print(f"Invoked: {tlsf_key} trial {trial}")

print("\nAll invocations sent!")
print(f"Monitor: aws s3 ls s3://{BUCKET}/results/ | wc -l")
