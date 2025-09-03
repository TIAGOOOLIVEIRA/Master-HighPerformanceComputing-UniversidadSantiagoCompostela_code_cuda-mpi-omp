import json
import boto3
import os

ssm = boto3.client("ssm")

def lambda_handler(event, context):
    # SNS passes messages inside "Records"
    record = event["Records"][0]
    message = json.loads(record["Sns"]["Message"])

    job_script = message.get("jobScript", "default.sh")
    params = message.get("parameters", "")

    # Head node instance ID (tag your head node and look up dynamically if needed)
    head_node_id = os.environ["HEAD_NODE_INSTANCE_ID"]

    # Command to submit the job
    command = f"sbatch {job_script} {params}"

    response = ssm.send_command(
        InstanceIds=[head_node_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command]}
    )

    return {
        "status": "submitted",
        "job_script": job_script,
        "params": params,
        "ssm_response": response
    }
