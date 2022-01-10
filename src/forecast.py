# forecast.py
import sys
import os

import ui.forecast_ui

forecast = ui.forecast_ui.Ui()

print(forecast.dataset_path)
print(forecast.ts_number)