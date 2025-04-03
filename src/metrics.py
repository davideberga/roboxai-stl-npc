import copy
from typing import Dict, List
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
    mean_velocity = sum(velocity_for_epi.values()) / len(velocity_for_epi.values())
    
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
def generate_markdown_table(title: str, column_names: List, methods: Dict) -> str:
    mdFile = MdUtils(file_name="temp")
    n_cols = len(column_names)
    n_rows = len(methods.keys()) + 1
    
    mdFile.new_header(level=3, title=title, add_table_of_contents='n')

    table_data = column_names
    
    for method, path in methods.items():
        print(f'------------------------------------ {method} ------------------------------------')
        episodes = read_npz(path)
        metrics = calculate_metrics(episodes)
        row = [
            method,
            str(metrics[0]),  # N_Goals_Reached
            str(metrics[1]),  # Mean Battery %
            str(metrics[2]),  # Battery std_dev
            str(metrics[3]),  # Mean Velocity
            str(metrics[4]),  # Velocity std_dev
            str(metrics[5]),  # Mean Abs Delta Velocity
            str(metrics[6]),  # Safety %
            str(metrics[7])   # Low Battery %
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
        'OUR': 'STL/test-result/our.result.npz'
    }
    methods_graphics = {
        'Paper': 'STL/test-result/paper-figure.result.npz',
        'OUR': 'STL/test-result/our-figure.result.npz'
    }
    columns = ['Method', 'N_Goals_Reached', 'Mean Battery %', 'Battery std_dev', 
               'Mean Velocity', 'Velocity std_dev', 'Mean Abs Delta Velocity', 
               'Safety %', 'Low Battery %']


    table_unity_md = generate_markdown_table("Test in unity", copy.deepcopy(columns), methods_unity)
    table_graphics_md = generate_markdown_table("Test in graphical env", copy.deepcopy(columns), methods_graphics)


    existing_md_file = "../README.md"
    combined_tables = table_graphics_md  + "\n\n" +  table_unity_md
    
    update_md_file(existing_md_file, combined_tables)