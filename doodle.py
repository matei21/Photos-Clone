from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / 'modeling' / 'inscrieri2026.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(df['Timestamp'], df.index)
plt.xlabel('Datetime')
plt.ylabel('Number of Registrations')
plt.title('Registrations Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()