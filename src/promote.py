import argparse
import os
import mlflow
from mlflow import MlflowClient

p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
p.add_argument('--alias', default='champion')
args = p.parse_args()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
client = MlflowClient()

client.set_registered_model_alias('ckd-classifier', args.alias, args.version)
print(f'Alias {args.alias} -> version {args.version}')