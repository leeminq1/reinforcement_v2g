import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_result_directories():
    """결과 저장을 위한 디렉토리 구조 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 기본 디렉토리 구조 생성
    base_dir = "results"
    csv_dir = os.path.join(base_dir, "csv", timestamp)
    png_dir = os.path.join(base_dir, "png", timestamp)
    
    # 디렉토리 생성
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    return csv_dir, png_dir

def analyze_results(csv_file):
    # 결과 저장 디렉토리 생성
    csv_dir, png_dir = create_result_directories()
    
    # CSV 파일 로드
    df = pd.read_csv(csv_file)
    
    # 1. 에피소드별 보상 추이
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='time', y='reward', hue='episode')
    plt.title('에피소드별 시간당 보상')
    plt.xlabel('시간')
    plt.ylabel('보상')
    plt.savefig(os.path.join(png_dir, 'rewards_by_episode.png'))
    plt.close()
    
    # 2. 건물 부하와 총 부하 비교 (모든 에피소드)
    plt.figure(figsize=(15, 8))
    for ep in df['episode'].unique():
        ep_data = df[df['episode'] == ep]
        plt.plot(ep_data['time'], ep_data['building_load'], 
                label=f'건물 부하 (EP {ep})', alpha=0.3)
        plt.plot(ep_data['time'], ep_data['total_load'], 
                label=f'총 부하 (EP {ep})', alpha=0.3)
    plt.axhline(y=400, color='r', linestyle='--', label='피크 임계값')
    plt.title('시간별 부하 변화')
    plt.xlabel('시간')
    plt.ylabel('부하 (kW)')
    plt.legend()
    plt.savefig(os.path.join(png_dir, 'loads_comparison.png'))
    plt.close()
    
    # 3. SMP 가격과 충방전 패턴
    plt.figure(figsize=(15, 12))
    
    # 3-1. SMP 가격
    plt.subplot(2, 1, 1)
    for ep in df['episode'].unique():
        ep_data = df[df['episode'] == ep]
        plt.plot(ep_data['time'], ep_data['smp_price'], 
                label=f'Episode {ep}', alpha=0.7)
    plt.title('시간별 SMP 가격')
    plt.xlabel('시간')
    plt.ylabel('가격 (원/kWh)')
    plt.legend()
    
    # 3-2. EV 충방전 패턴 (마지막 에피소드)
    plt.subplot(2, 1, 2)
    last_ep = df['episode'].max()
    last_ep_data = df[df['episode'] == last_ep]
    
    ev_columns = [col for col in df.columns if 'charging_rate' in col]
    for col in ev_columns:
        plt.plot(last_ep_data['time'], last_ep_data[col], 
                label=f'EV {col.split("_")[0]}', alpha=0.7)
    plt.title(f'EV 충방전 패턴 (Episode {last_ep})')
    plt.xlabel('시간')
    plt.ylabel('충방전량 (kW)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, 'smp_and_charging.png'), bbox_inches='tight')
    plt.close()
    
    # 4. SOC 변화 (마지막 에피소드)
    plt.figure(figsize=(15, 8))
    soc_columns = [col for col in df.columns if 'soc' in col]
    for col in soc_columns:
        plt.plot(last_ep_data['time'], last_ep_data[col], 
                label=f'EV {col.split("_")[0]}', alpha=0.7)
    plt.title(f'EV SOC 변화 (Episode {last_ep})')
    plt.xlabel('시간')
    plt.ylabel('SOC (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, 'soc_changes.png'), bbox_inches='tight')
    plt.close()
    
    # 5. 통계 정보 출력
    print("\n=== 학습 결과 분석 ===")
    print(f"총 에피소드 수: {df['episode'].nunique()}")
    print(f"평균 보상: {df.groupby('episode')['reward'].sum().mean():.2f}")
    print(f"피크 발생 횟수: {len(df[df['total_load'] > 400])}")
    print("\n에피소드별 총 보상:")
    print(df.groupby('episode')['reward'].sum())
    
    # 원본 데이터 저장
    df.to_csv(os.path.join(csv_dir, 'training_results.csv'), index=False)
    
    # 통계 정보 CSV 파일 생성
    stats_df = pd.DataFrame({
        'episode': df['episode'].unique(),
        'total_reward': df.groupby('episode')['reward'].sum(),
        'peak_occurrences': df[df['total_load'] > 400].groupby('episode').size(),
        'avg_building_load': df.groupby('episode')['building_load'].mean(),
        'avg_total_load': df.groupby('episode')['total_load'].mean(),
        'avg_smp_price': df.groupby('episode')['smp_price'].mean()
    })
    stats_df.to_csv(os.path.join(csv_dir, 'training_statistics.csv'))
    
    # 결과 저장 위치 출력
    print(f"\n결과가 저장된 위치:")
    print(f"CSV 파일: {csv_dir}")
    print(f"그래프: {png_dir}") 