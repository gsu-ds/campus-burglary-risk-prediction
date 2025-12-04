Single Weather pulled or NPU specific?
- Probably single weather for simplicity


Claude rec:
# Additional temporal feature (captures dusk/dawn crime patterns)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)