import pandas as pd
import numpy as np
import re

# RPL Data Extraction Functions
def get_radio_data(file_path, first_ID):
    """
    Reads radio data from a CSV file and processes it.
    
    Args:
        file_path (str): Path to the CSV file containing radio data.
        first_ID (int): The starting ID for the nodes.
        
    Returns:
        pd.DataFrame: Processed DataFrame with radio data.
    """
    
    # extract destination(s) IDs from the capture file
    def get_dsts(line):
        return(line.split(' ')[0].strip().split('\t')[2].split(','))
    
    # extract timestamp from the capture file
    def get_timestamp(line):
        return line.split(' ')[0].strip().split('\t')[0]
    
    # extract source ID from the capture file
    def get_src(line):
        return line.split(' ')[0].strip().split('\t')[1]
    
    # extract RPL message type from the capture file
    def get_rplcode(line):
        match = re.search(r'RPL\s+(\w+)', line)
        if match:
            return(match.group(1))
    
   
   # extract all the features
    def parse_radio_data(file):
        with open(file, 'r') as file:
            lines = file.readlines()
            print(len(lines))
        radio_data_list = []
        for line in lines:
            if 'RPL' in line.split(' '):
            
                for dst in get_dsts(line):
                    radio_data_list.append({
                    'time_stamp': get_timestamp(line),
                    'source': get_src(line),
                    'destination': dst,
                    'RPL_message': get_rplcode(line)
                    })
            else:
                continue
        df = pd.DataFrame(radio_data_list)
        return df
   
    # create DataFrame for raw data
    df_raw = parse_radio_data(file_path)

    # sort DataFrame by source ID
    df_raw_sorted = df_raw.sort_values(by='source') 

    # create the final DataFrame
    df_radio = pd.DataFrame(columns=['ID', 'DIS_sent', 'DIS_received', 'DIO_sent', 'DIO_received', 'DAO_sent', 'DAO_received'])
    
    # adding unique IDs
    IDs = pd.DataFrame(np.sort(np.int32(df_raw_sorted['source'].unique())), columns=['ID'])
    df_radio['ID'] = np.arange(first_ID, df_raw_sorted['source'].nunique() + first_ID)

    # addding the total number of DIS messages sent by each node
    def get_dis_sent(item):
        return df_raw_sorted[(df_raw_sorted['source'] == str(item)) & (df_raw_sorted['RPL_message'] == 'DIS')].count()['RPL_message'] 

    df_radio['DIS_sent'] = IDs['ID'].apply(get_dis_sent)   
    
    # addding the total number of DIS messages received by each node
    def get_dis_received(item):
        return df_raw_sorted[(df_raw_sorted['destination'] == str(item)) & (df_raw_sorted['RPL_message'] == 'DIS')].count()['RPL_message']
    
    df_radio['DIS_received'] = IDs['ID'].apply(get_dis_received)

    # addding the total number of DIO messages sent by each node
    def get_dio_sent(item):
        return df_raw_sorted[(df_raw_sorted['source'] == str(item)) & (df_raw_sorted['RPL_message'] == 'DIO')].count()['RPL_message']
    
    df_radio['DIO_sent'] = IDs['ID'].apply(get_dio_sent)

    # addding the total number of DIO messages received by each node
    def get_dio_received(item):
        return df_raw_sorted[(df_raw_sorted['destination'] == str(item)) & (df_raw_sorted['RPL_message'] == 'DIO')].count()['RPL_message']
    
    df_radio['DIO_received'] = IDs['ID'].apply(get_dio_received)

    # addding the total number of DAO messages sent by each node
    def get_dao_sent(item):
        return df_raw_sorted[(df_raw_sorted['source'] == str(item)) & (df_raw_sorted['RPL_message'] == 'DAO')].count()['RPL_message']
    
    df_radio['DAO_sent'] = IDs['ID'].apply(get_dao_sent)

    # addding the total number of DAO messages received by each node
    def get_dao_received(item):
        return df_raw_sorted[(df_raw_sorted['destination'] == str(item)) & (df_raw_sorted['RPL_message'] == 'DAO')].count()['RPL_message']
    
    df_radio['DAO_received'] = IDs['ID'].apply(get_dao_received)

    return df_radio


