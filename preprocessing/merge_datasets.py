# Merge Datasets
def create_final_dataset(radio_file, power_file, first_ID=1, attackers_ID=[0]):
    """
    Creates a final dataset by merging radio and power data.
    
    Args:
        radio_file (str): Path to the CSV file containing radio data.
        power_file (str): Path to the CSV file containing power data.
        first_ID (int): Starting ID for the nodes.
        attackers_ID (list): List of attacker node IDs.

    Returns:
        pd.DataFrame: Merged DataFrame with radio and power data.
    """
    
    df_radio = get_radio_data(radio_file, first_ID )
    df_power = get_power_data(power_file, first_ID)
    
    # merge DataFrames
    df_final = pd.merge(df_radio, df_power, on='ID')
    df_final['label'] = df_final['ID'].apply(lambda x: 1 if x in attackers_ID else 0)
    return df_final 