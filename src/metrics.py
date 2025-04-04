import copy
from typing import Dict, List
import numpy as np
from mdutils.mdutils import MdUtils
import statistics
import torch
from STL.alg.RoverSTL import RoverSTL
from STL import config

chunk_size = 5

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

# Funzione per dividere gli episodi in chunk da 10 step
def divide_episodes(episodes, chunk_size=10):
    result = []
    for i, epi in enumerate(episodes):
        epi = np.array(epi)
        epi = epi[:, np.r_[0:11, 17, 18]]  # Seleziona solo i primi 11 valori e il 17° e 18° = 13 valori totali
        num_steps = len(epi)
        num_full_chunks = num_steps // chunk_size # 
        last_steps = epi[-(num_full_chunks * chunk_size):]
        chunks = last_steps.reshape(-1, chunk_size, epi.shape[1]) if num_full_chunks > 0 else []

        for c in chunks:
            result.append(c)
        
    return result

def calculate_accuracy(result, rover_stl):
    stl, _, _, _, _, _ = rover_stl.generateSTL(steps_ahead=chunk_size, battery_limit=2.0)
    accuracy = []
    
    for i, chunk in enumerate(result):
        t = torch.tensor(chunk).float().to(rover_stl.device)
        t = t.unsqueeze(0)  # aggiunge una dimensione all'inizio
        accuracy.append((stl(t, rover_stl.smoothing_factor, d={"hard": True})[:, :1] >= 0).float())
        
    if len(accuracy) > 0:
        accuracy = torch.cat(accuracy, dim=0)
        acc_avg = torch.mean(accuracy)
    else: 
        print("La lista accuracy è vuota!")
        acc_avg = torch.tensor(0.0)
    
    return acc_avg * 100
    

def calculate_metrics(episodes, rover_stl, method_name):
    battery_for_epi = {}
    velocity_for_epi = {}
    goal_for_epi = []
    velocity_for_delta = []
    min_radar_list = []
    collision_list = []
    low_battery_list = []
    mean_battery_list = []
    mean_velocity_list = []
    min_lidar_list = []
    collision = 0
    total_len_episodes = 0
    safe_threshold = 0.15 
    

    no_episodes = len(episodes)

    for i, epi in enumerate(episodes):
        goal_for_epi.append(np.max(epi[:,21]))
        collision_list.append(np.max(epi[:,22]))
        low_battery_list.append(np.max(epi[:,23]))
        
        mean_battery_list.append(np.sum(epi[:,17]) / len(epi))   
        mean_velocity_list.append(np.sum(epi[:,19]) / len(epi)) 

        min_lidar_list.append(np.min(epi[:,0:7]))

        temp_list = []
        for step in epi:
            total_len_episodes += 1
            temp_list.append(step[19])
            min_radar_list.append(min(step[0:7]))
            
        velocity_for_delta.append(temp_list)

    perc_goals = (np.sum(goal_for_epi) / no_episodes) * 100 

    # Quante volte c'è stata collisone
    collision = (np.sum(collision_list) / no_episodes) * 100
    
    # Quante volte la batteria si è scaricata
    low_battery = (np.sum(low_battery_list) / no_episodes) * 100

    #print(np.mean(np.array(list(goal_for_epi.values()))))
   
    # Calcolo della media della percentuale di batteria per test
    perc_battery = ((np.sum(mean_battery_list) / no_episodes) * 100) / 5.0 # 5.0 è la capacità massima della batteria
    # Calcolo della deviazione standard della batteria
    std_dev_battery = np.std(mean_battery_list) 

     # Calcolo della media della velocità per test
    mean_velocity = np.sum(mean_velocity_list) / no_episodes
    # Calcolo della deviazione standard della velocità
    std_dev_velocity = np.std(mean_velocity_list)

    # Calcolo delta velocità per ogni episodio separatamente
    delta_v = [np.diff(episode) for episode in velocity_for_delta]

    # Calcolo della media assoluta del delta velocità, ignorando episodi vuoti
    mean_abs_delta_v = np.mean([np.mean(np.abs(ep)) for ep in delta_v if len(ep) > 0])

    # Calcolo della percentuale di volte che il lidar in min_radar_list è maggiore di 0.15
    safety = np.sum(np.array(min_lidar_list) > safe_threshold)

    # if method_name == 'DQN':
    #     accuracy = torch.tensor(0.0)
    # else:
    # Rules accuracy
    result = divide_episodes(episodes, chunk_size)
    accuracy = calculate_accuracy(result, rover_stl)
    
    b_correlations = []
    for epi in episodes:
        np_episode = np.array(epi)
        mask = np_episode[:, 17] < 2
        battery_filtered = np_episode[mask, 17]
        distance_filtered = np_episode[mask, 10]
        if len(battery_filtered) > 1 and np.std(battery_filtered) > 0 and np.std(distance_filtered) > 0:
            corr = np.corrcoef(battery_filtered, distance_filtered)[0, 1]
        else:
            continue
            
        b_correlations.append(np.nan_to_num(corr))
    
    battery_corr = np.mean(np.array(b_correlations))

    low_battery = round(low_battery, 2)
    safety = round(safety, 2)
    mean_velocity = round(mean_velocity, 2)
    mean_abs_delta_v = round(mean_abs_delta_v, 2)
    std_dev_velocity = round(std_dev_velocity, 2)
    std_dev_battery = round(std_dev_battery, 2)
    perc_battery = round(perc_battery, 2)
    perc_goals = round(perc_goals, 2)
    collision = round(collision, 2)
    battery_corr = round(battery_corr, 2)
    
    # Stampa dei risultati
    print('------------------------------------------------------------')
    print(f"Goal Percentage: {perc_goals}%")
    print(f"Battery Percentage: {perc_battery}%")
    print(f"Battery std_dev: {std_dev_battery}")
    print(f"Mean Velocity: {mean_velocity}")
    print(f"Velocity std_dev: {std_dev_velocity}")
    print(f"Mean Abs Delta Velocity: {mean_abs_delta_v}")
    print(f"Safety: {safety}%")
    print(f"Low battery: {low_battery}")
    print(f"Accuracy: {accuracy}%")
    print(f"Battery correlation: {battery_corr}")
    print(f"Collision: {collision}")
    print('------------------------------------------------------------')

    return perc_goals, perc_battery, std_dev_battery, mean_velocity, std_dev_velocity, mean_abs_delta_v, safety, low_battery, accuracy, battery_corr, collision

# Funzione per creare una tabella Markdown
def generate_markdown_table(title: str, column_names: List, methods: Dict) -> str:
    mdFile = MdUtils(file_name="temp")
    n_cols = len(column_names)
    n_rows = len(methods.keys()) + 1
    
    mdFile.new_header(level=3, title=title, add_table_of_contents='n')

    args = config.parse_args()
    rover_stl = RoverSTL(None, args)

    table_data = column_names
    
    for method, path in methods.items():
        print(f'------------------------------------ {method} ------------------------------------')
        episodes = read_npz(path)
        metrics = calculate_metrics(episodes, rover_stl, method)

        row = [
            method,
            str(metrics[0]),  # N_Goals_Reached
            str(metrics[1]) + ' ± ' + str(metrics[2]),  # Mean Battery % and # Battery std_dev
            str(metrics[3]) + ' ± ' + str(metrics[4]),  # Mean Velocity and # Velocity std_dev
            str(metrics[5]),  # Mean Abs Delta Velocity
            str(metrics[6]),  # Safety %
            str(metrics[7]),   # Low Battery %
            str(round(metrics[8].item(), 2)),  # Accuracy %
            str(metrics[9]),   # Battery correlation
            str(metrics[10])  # Collision %
        ]
        table_data.extend(row)

    mdFile.new_table(columns=n_cols, rows=n_rows, text=table_data, text_align="center")
    table_md = mdFile.get_md_text()
    return table_md

def update_md_file(existing_md_path: str, new_content: str, start_marker: str="<!-- START TABLES -->", end_marker: str="<!-- END TABLES -->"):
    import re
    try:
        with open(existing_md_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except FileNotFoundError:
        file_content = ""
        
    pattern = re.compile(
        re.escape(start_marker) + ".*?" + re.escape(end_marker),
        re.DOTALL
    )
    new_section = f"{start_marker}\n{new_content}\n{end_marker}"

    if re.search(pattern, file_content):
        file_content = re.sub(pattern, new_section, file_content)
    else:
        file_content += "\n\n" + new_section

    with open(existing_md_path, "w", encoding="utf-8") as f:
        f.write(file_content)

if __name__ == "__main__":
    methods_unity = {
        'Paper': 'STL/test-result/paper.result.npz',
        'DQN': 'DQN/test-result/dqn.result.npz',
        'OUR': 'STL/test-result/our.result.npz',
        'No avoid rule': 'STL/test-result/no_avoid.result.npz'
    }
    methods_graphics = {
        'Paper': 'STL/test-result/paper-figure.result.npz',
        'OUR': 'STL/test-result/our-figure.result.npz',
        'No avoid rule': 'STL/test-result/no_avoid-figure.result.npz',
    }
    columns = ['Method', 'N_Goals_Reached', 'Mean Battery %',
               'Mean Velocity', 'Mean Abs Delta Velocity', 
               'Safety %', 'Low Battery %', 'Accuracy %', 'Battery correlation', 'Collision %']


    table_unity_md = generate_markdown_table("Test in unity", copy.deepcopy(columns), methods_unity)
    table_graphics_md = generate_markdown_table("Test in graphical env", copy.deepcopy(columns), methods_graphics)


    existing_md_file = "../README.md"
    combined_tables = table_graphics_md  + "\n\n" +  table_unity_md
    
    update_md_file(existing_md_file, combined_tables)