import json
import urllib.request
from pathlib import Path

body = Path('sample_request.json').read_bytes()
req = urllib.request.Request('http://127.0.0.1:8000/predict', data=body, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.load(r)
except Exception as e:
    print(f"ОШИБКА ПОДКЛЮЧЕНИЯ К API: {e}")
    print("Убедись, что контейнер запущен командой: sudo docker ps")
    exit(1)

expected = json.loads(Path('expected_predictions.json').read_text())
actual_preds = result.get('predictions')

if actual_preds == expected:
    print("PASS: container predictions match exported model")
    print(json.dumps(result, indent=2))
else:
    print("FAIL: predictions do not match!")
    print(f"Ожидалось: {expected}")
    print(f"Получено: {actual_preds}")