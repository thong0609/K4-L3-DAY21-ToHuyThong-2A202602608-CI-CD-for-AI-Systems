# AWS deployment for this lab

Region: ap-southeast-2. Bucket: income-lab-tohuythong-20261008.
EC2: i-019e5c318d341fce9, 13.211.138.184, user ubuntu.
Security group: sg-067a82f5775c9ce91.

GitHub Actions secrets: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
ARTIFACT_BUCKET, SERVER_HOST, SERVER_USER, SERVER_SSH_KEY.
Repository variable: AWS_REGION=ap-southeast-2.

Local PowerShell (activate .venv first):

```powershell
$env:AWS_PROFILE = "income-lab"
$env:AWS_DEFAULT_REGION = "ap-southeast-2"
dvc pull
python -m pytest tests/ -v
python src/train.py
```

The profile in .dvc/config.local is local only. CI uses environment credentials.
Data in dvc/ is addressed by hashes; original CSV filenames are in the .dvc pointers.

EC2 uses /home/ubuntu/income-api/.venv, Python 3.10, scikit-learn 1.4.2,
pandas 2.2.2, joblib 1.4.2, fastapi 0.111.0, uvicorn 0.29.0 and boto3.
AWS credentials remain in /home/ubuntu/.aws/credentials, outside the repository.
The service runs as ubuntu and downloads the approved model on every restart.

The pipeline runs Unit Test -> Train -> Quality Gate -> Release. Training saves
model/report as GitHub artifacts. Only Release uploads the model to S3 after
positive-class F1 >= 0.65, copies the API and service, and checks health.
SSH verifies the EC2 host key against deploy/known_hosts (retrieved over the
existing trusted SSH connection). Update this file if replacing the instance.
If EC2 public IP changes, update SERVER_HOST and the host in deploy/known_hosts.

The existing security group permits ports 22 and 8080 from all IPv4 addresses.
No firewall rules were changed by this setup. Restrict SSH sources when replacing
this lab setup with a persistent deployment.

API checks (PowerShell):

```powershell
Invoke-RestMethod http://13.211.138.184:8080/healthz
Invoke-RestMethod http://13.211.138.184:8080/score -Method Post -ContentType 'application/json' -Body '{"features":[28,2,14,2,11,0,1,0,0,45]}'
```
