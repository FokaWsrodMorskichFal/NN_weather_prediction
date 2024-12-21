import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
'''
# Read data from file
file_path = './data/city_attributes.csv'
data = pd.read_csv(file_path)

# Extract latitude and longitude
latitudes = data['Latitude']
longitudes = data['Longitude']
city_names = data['City']

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(longitudes, latitudes, c='red', marker='o', edgecolor='black', alpha=0.7)

# Annotate each city
for i, city in enumerate(city_names):
    plt.text(longitudes[i] + 0.2, latitudes[i] + 0.2, city, fontsize=8)

# Add labels and title
plt.title('Cities on Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Show the map
plt.show()
'''
'''
dropout_rate = 0.3

tab = np.array([1, 2, 3, 4, 3, 2, 1,])

mask = np.random.rand(*tab.shape) > dropout_rate

print(tab)
print(mask)
print(tab*mask)
'''

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[0:6])

'''
days = [i for i in range(365)]
shifted_days = [((i-135)%365) for i in range(365)]
print(shifted_days)
time_angle = [i*2*math.pi/364 for i in shifted_days]
print(time_angle)
encoding = [[np.cos(time), np.sin(time)] for time in time_angle]
print(encoding)
'''