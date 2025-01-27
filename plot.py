import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytaj dane z pliku CSV
df = pd.read_csv("./results/final_results.csv")

# Ustawienie szerokości słupków i pozycji
bar_width = 0.35
x = np.arange(len(df["City"]))

# Tworzenie pierwszego wykresu - Temperatura
plt.figure(figsize=(15, 8))
plt.bar(x - bar_width / 2, df["temperature accuracy small model"], bar_width, label="Small Model")
plt.bar(x + bar_width / 2, df["temperature accuracy big model"], bar_width, label="Big Model")

# Ustawienia osi i etykiet dla temperatury
plt.xticks(x, df["City"], rotation=90)
plt.xlabel("City")
plt.ylabel("Temperature Accuracy")
plt.title("Temperature Accuracy for Small and Big Models")
plt.legend()

# Wyświetlenie wykresu dla temperatury
plt.tight_layout()
plt.show()

# Tworzenie drugiego wykresu - Wiatr
plt.figure(figsize=(15, 8))
plt.bar(x - bar_width / 2, df["wind accuracy small model"], bar_width, label="Small Model")
plt.bar(x + bar_width / 2, df["wind accuracy big model"], bar_width, label="Big Model")

# Ustawienia osi i etykiet dla wiatru
plt.xticks(x, df["City"], rotation=90)
plt.xlabel("City")
plt.ylabel("Wind Accuracy")
plt.title("Wind Accuracy for Small and Big Models")
plt.legend()

# Wyświetlenie wykresu dla wiatru
plt.tight_layout()
plt.show()
