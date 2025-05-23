# -*- coding: utf-8 -*-
from environment import V2GEnvironment
from dqn_agent import DQNAgent
from visualization import V2GVisualization
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.font_manager as fm
import pandas as pd
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def train_dqn():
    # Initialize environment and agent
    num_vehicles = 20
    env = V2GEnvironment(num_vehicles)
    
    # 상태 크기 계산: (차량당 4개 상태값) * 차량수 + 3(전기가격, 건물부하, 시간)
    state_size = num_vehicles * 4 + 3
    action_size = num_vehicles
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Training parameters
    episodes = 60  # 에피소드 수 감소
    target_update_frequency = 5
    rewards_history = []
    
    # 프레임 간격 설정
    frame_delay = 0.1  # 프레임 간격 감소
    
    # 결과 저장을 위한 데이터 구조
    all_episode_data = []
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            episode_data = []  # 현재 에피소드의 데이터 저장
            
            while not done:
                # Select and execute action
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # 현재 시간스텝의 데이터 수집
                step_data = {
                    'episode': episode,
                    'time': env.current_time,
                    'building_load': env.current_load,
                    'total_load': env.current_load + sum([v['charging_rate'] for v in env.vehicles.values() if v['present']]),
                    'smp_price': env.current_price,
                    'reward': reward
                }
                
                # 각 차량의 상태 저장
                for i in range(num_vehicles):
                    vehicle = env.vehicles[i]
                    if vehicle['present']:
                        step_data.update({
                            f'ev{i}_present': 1,
                            f'ev{i}_soc': vehicle['soc'],
                            f'ev{i}_charging_rate': vehicle['charging_rate'],
                            f'ev{i}_arrival_time': vehicle['arrival_time'],
                            f'ev{i}_departure_time': vehicle['departure_time']
                        })
                    else:
                        step_data.update({
                            f'ev{i}_present': 0,
                            f'ev{i}_soc': 0,
                            f'ev{i}_charging_rate': 0,
                            f'ev{i}_arrival_time': -1,
                            f'ev{i}_departure_time': -1
                        })
                
                episode_data.append(step_data)
                
                # Scale rewards
                reward = reward / 1000.0
                
                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                state = next_state
                
                # Update visualization
                if not env.visualization.update(env.vehicles, 
                                             env.current_time,
                                             env.current_price,
                                             total_reward,
                                             episode,
                                             env.current_load):
                    return agent
                
                # 프레임 간격 적용
                time.sleep(frame_delay)
            
            # Update target network
            if episode % target_update_frequency == 0:
                agent.update_target_model()
            
            rewards_history.append(total_reward)
            all_episode_data.extend(episode_data)
            
            # Print progress
            print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # 결과를 DataFrame으로 변환하고 CSV로 저장
        results_df = pd.DataFrame(all_episode_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'training_results_{timestamp}.csv', index=False)
        
        # Visualize learning progress
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_history)
        plt.title('Learning Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('learning_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    finally:
        env.visualization.close()
    
    return agent

if __name__ == "__main__":
    trained_agent = train_dqn()
    
    # 학습 완료 후 결과 분석 실행
    print("\n=== 학습 완료, 결과 분석 시작 ===")
    from analyze_results import analyze_results
    from glob import glob
    
    # 가장 최근의 결과 파일 찾기
    result_files = glob('training_results_*.csv')
    if result_files:
        latest_file = max(result_files)
        print(f"분석할 파일: {latest_file}")
        analyze_results(latest_file)
    else:
        print("결과 파일을 찾을 수 없습니다.") 