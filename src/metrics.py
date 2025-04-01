import numpy as np
from mdutils.mdutils import MdUtils
import statistics

# Funzione per leggere il file NPZ e restituisce gli episodi (100) con i relativi step
def read_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    data_episodes = data.files

    # Stampa le chiavi del file NPZ
    # print(f"data.files: {data_episodes}")
   
    # Determinare il numero di episodi (dovrebbe essere 100)
    tot_episodes = len(data[data_episodes[0]])
    #print(f"Numero di episodi per questo test: {tot_episodes}")

    episodes = data[data_episodes[0]]

    # for i,epi in enumerate(episodes):
    #     if i < 10 :
    #         print(epi[0])

    return episodes
    

def calculate_metrics(episodes):
    battery_for_epi = {}
    velocity_for_epi = {}
    goal_for_epi = []
    velocity_for_delta = []
    min_radar_list = []
    low_battery = 0
    total_len_episodes = 0
    safe_threshold = 0.15 # calcola in numero di volte che il min lidar in min_radar_list è minore di 0.15
    
    
    no_episodes = len(episodes)

    for i, epi in enumerate(episodes):
        goal_for_epi.append(np.max(epi[:,21]))
        
        mean_battery = sum(step[17] for step in epi) / len(epi)  # Media della batteria per episodio
        mean_velocity = sum(step[19] for step in epi) / len(epi)  # Media della velocità per episodio
        level_battery = (100 * mean_battery) / 5.0  # Converti in percentuale
        battery_for_epi[i] = level_battery 
        velocity_for_epi[i] = mean_velocity

        low_battery += np.max(epi[:,23])

        temp_list = []
        for step in epi:
            total_len_episodes += 1
            temp_list.append(step[19])
            min_radar_list.append(min(step[0:7]))
            
        velocity_for_delta.append(temp_list)

    perc_goals = (np.sum(goal_for_epi) / no_episodes) * 100 

    #print(np.mean(np.array(list(goal_for_epi.values()))))
   
    # Calcolo della media della percentuale di batteria per test
    perc_battery = sum(battery_for_epi.values()) / len(battery_for_epi)

    # Calcolo della media della velocità per test
    mean_velocity = sum(velocity_for_epi.values()) / len(velocity_for_epi)
    
    # Calcolo della deviazione standard della batteria
    std_dev_battery = statistics.stdev(battery_for_epi.values())  

    # Calcolo della deviazione standard della velocità
    std_dev_velocity = statistics.stdev(velocity_for_epi.values()) 

    # Calcolo delta velocità per ogni episodio separatamente
    delta_v = [np.diff(episode) for episode in velocity_for_delta]

    # Calcolo della media assoluta del delta velocità, ignorando episodi vuoti
    mean_abs_delta_v = np.mean([np.mean(np.abs(ep)) for ep in delta_v if len(ep) > 0])

    safety = np.mean(min_radar_list) * 100

    low_battery = round(low_battery, 2)
    safety = round(safety, 2)
    mean_velocity = round(mean_velocity, 2)
    mean_abs_delta_v = round(mean_abs_delta_v, 2)
    std_dev_velocity = round(std_dev_velocity, 2)
    std_dev_battery = round(std_dev_battery, 2)
    perc_battery = round(perc_battery, 2)
    perc_goals = round(perc_goals, 2)

    print('------------------------------------------------------------')
    print(f"Goal Percentage: {perc_goals}%")
    print(f"Battery Percentage: {perc_battery}%")
    print(f"Battery std_dev: {std_dev_battery}")
    print(f"Mean Velocity: {mean_velocity}")
    print(f"Velocity std_dev: {std_dev_velocity}")
    print(f"Mean Abs Delta Velocity: {mean_abs_delta_v}")
    print(f"Safety: {safety}%")
    print(f"Low battery: {low_battery}")
    print('------------------------------------------------------------')

    return perc_goals, perc_battery, std_dev_battery, mean_velocity, std_dev_velocity, mean_abs_delta_v, safety, low_battery

# Funzione per creare una tabella Markdown
def create_markdown_table(name_md_file, column_names):
    mdFile = MdUtils(file_name=name_md_file, title="Results") 
    
    # Definire i metodi analizzati
    methods = ['Paper', 'DQN', 'OUR']

    # Preparare i dati della tabella
    table_data = column_names  # Intestazione
    

    for method in methods:
        if method == 'Paper':
            print('------------------------------------PAPER------------------------------------')
            episodes = read_npz("STL/test-result/paper.result.npz")
        elif method == 'DQN':
            print('------------------------------------DQN------------------------------------')
            episodes = read_npz("DQN/test-result/dqn.result.npz")
        elif method == 'OUR':
            print('------------------------------------OUR------------------------------------')
            episodes = read_npz("STL/test-result/our.result.npz")

        perc_goals, perc_battery, std_dev_battery, mean_velocity, std_dev_velocity, mean_abs_delta_v, safety, low_battery = calculate_metrics(episodes)
    
        
        table_data.extend([
            method,  # Metodo
            str(perc_goals),  # Percentuale goal raggiunti
            str(perc_battery),  # Percentuale batteria
            str(std_dev_battery),  # Deviazione standard batteria
            str(mean_velocity),  # Velocità media
            str(std_dev_velocity),  # Deviazione standard velocità
            str(mean_abs_delta_v),  # Media assoluta delta velocità
            str(safety),  # Sicurezza
            str(low_battery)  # Percentuale batteria carica
        ])
    

    # Creare la tabella Markdown
    mdFile.new_table(columns=9, rows=len(methods) + 1, text=table_data, text_align="center")
    
    # Salvare il file Markdown
    mdFile.create_md_file()


create_markdown_table("Results", ['Method', 'N_Goals_Reached', 'Mean Battery %', 'Battery std_dev', 'Mean Velocity', 'Velocity std_dev', 'Mean Abs Delta Velocity', 'Safety %', 'Low Battery %'])
