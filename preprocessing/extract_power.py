import pandas as pd
import numpy as np
import re

# Power Data Extraction Functions
# === Contiki parameters (Sky mote typical values) ===
RTIMER_SECOND = 32768  # ticks per second
VOLTAGE = 3.0          # supply voltage in Volts

# Current consumption in Amps
I_CPU = 1.8e-3
I_LPM = 0.0545e-3
I_TX  = 17.4e-3
I_RX  = 18.8e-3

# === Function to calculate energies (mJ) and power (mW) ===
def calculate_energy_and_power(row):
    # Convert ticks → seconds
    t_cpu = row['CPU'] / RTIMER_SECOND
    t_lpm = row['LPM'] / RTIMER_SECOND
    t_tx  = row['TX']  / RTIMER_SECOND
    t_rx  = row['RX']  / RTIMER_SECOND
    t_total = row['TS'] / RTIMER_SECOND

    # Energy in mJ = T * I * V * 1000
    e_cpu = t_cpu * I_CPU * VOLTAGE * 1000
    e_lpm = t_lpm * I_LPM * VOLTAGE * 1000
    e_tx  = t_tx  * I_TX  * VOLTAGE * 1000
    e_rx  = t_rx  * I_RX  * VOLTAGE * 1000
    e_total = e_cpu + e_lpm + e_tx + e_rx

    # Average power in mW = Energy (mJ) / Total time (s)
    p_avg = e_total / t_total if t_total > 0 else 0.0

    return pd.Series({
        'E_CPU_mJ': e_cpu,
        'E_LPM_mJ': e_lpm,
        'E_TX_mJ': e_tx,
        'E_RX_mJ': e_rx,
        #'E_Total_mJ': e_total,
        'Power_mW': p_avg
    })

# === Function to get & process energy data ===
def get_power_data(file, first_ID):
    with open(file, 'r') as file:
        lines = file.readlines()
    power_data_list = []
    for line in lines:
        if 'ID:0' in line:
            pattern = r"ID:(?P<id>\d+)\s+ID:0\s+TS:(?P<ts>\d+)\s+CPU:(?P<cpu>\d+)\s+LPM:(?P<lpm>\d+)\s+TX:(?P<tx>\d+)\s+RX:(?P<rx>\d+)"            
            match = re.search(pattern, line)
            if match:
                power_data_list.append({
                    
                    'ID': int(match.group("id")) + first_ID - 1,
                    'TS': int(match.group("ts")),
                    'CPU': int(match.group("cpu")),
                    'LPM': int(match.group("lpm")),
                    'TX': int(match.group("tx")),
                    'RX': int(match.group("rx"))
                })
        else:
            continue

    df= pd.DataFrame(power_data_list).groupby('ID').max().reset_index()
    energy_power_df = df.apply(calculate_energy_and_power, axis=1)
    df = pd.concat([df, energy_power_df], axis=1)
    df = df.drop(columns=['TS', 'CPU', 'LPM', 'TX', 'RX'])

    return df