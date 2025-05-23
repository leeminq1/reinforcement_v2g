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

    # 20개 에피소드 단위로 그룹핑
    df['episode_group'] = df['episode'] // 20

    # 1. 에피소드 그룹별 보상 추이 (평균)
    plt.figure(figsize=(15, 8))
    grouped = df.groupby(['episode_group', 'time'], as_index=False)['reward'].mean()
    for group in grouped['episode_group'].unique():
        group_data = grouped[grouped['episode_group'] == group]
        plt.plot(group_data['time'], group_data['reward'], label=f'Group {int(group)*20}~{int(group)*20+19}')
    plt.title('20개 에피소드 그룹별 시간당 평균 보상')
    plt.xlabel('시간')
    plt.ylabel('평균 보상')
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    plt.xticks(range(0, 25, 1))
    plt.savefig(os.path.join(png_dir, 'rewards_by_episode.png'))
    plt.close()

    # 2. 건물 부하와 총 부하 비교 (20개 에피소드 그룹별 평균)
    plt.figure(figsize=(15, 8))
    grouped_load = df.groupby(['episode_group', 'time'], as_index=False)[['building_load', 'total_load']].mean()
    for group in grouped_load['episode_group'].unique():
        group_data = grouped_load[grouped_load['episode_group'] == group]
        plt.plot(group_data['time'], group_data['building_load'], label=f'건물 부하 (Group {int(group)*20}~{int(group)*20+19})', alpha=0.5)
        plt.plot(group_data['time'], group_data['total_load'], label=f'총 부하 (Group {int(group)*20}~{int(group)*20+19})', alpha=0.5)
    plt.axhline(y=400, color='r', linestyle='--', label='피크 임계값')
    plt.title('20개 에피소드 그룹별 시간별 평균 부하 변화')
    plt.xlabel('시간')
    plt.ylabel('부하 (kW)')
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    plt.xticks(range(0, 25, 1))
    plt.savefig(os.path.join(png_dir, 'loads_comparison.png'))
    plt.close()

    # 3. SMP 가격과 충방전 패턴 (20개 에피소드 그룹별 평균)
    plt.figure(figsize=(15, 12))
    # 3-1. SMP 가격
    plt.subplot(2, 1, 1)
    grouped_smp = df.groupby(['episode_group', 'time'], as_index=False)['smp_price'].mean()
    for group in grouped_smp['episode_group'].unique():
        group_data = grouped_smp[grouped_smp['episode_group'] == group]
        plt.plot(group_data['time'], group_data['smp_price'], label=f'Group {int(group)*20}~{int(group)*20+19}')
    plt.title('20개 에피소드 그룹별 시간별 평균 SMP 가격')
    plt.xlabel('시간')
    plt.ylabel('가격 (원/kWh)')
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    plt.xticks(range(0, 25, 1))

    # 3-2. EV 충방전 패턴 (EV별 subplot, 그룹별 평균)
    ev_columns = [col for col in df.columns if 'charging_rate' in col and 'ev' in col]
    fig, axes = plt.subplots(20, 1, figsize=(20, 2.5*20), sharex=True, sharey=True)
    for idx, col in enumerate(ev_columns):
        ax = axes[idx]
        grouped_ev = df.groupby(['episode_group', 'time'], as_index=False)[col].mean()
        for group in grouped_ev['episode_group'].unique():
            group_data = grouped_ev[grouped_ev['episode_group'] == group]
            ax.plot(group_data['time'], group_data[col], label=f'Group {int(group)*20}~{int(group)*20+19}')
        ax.set_title(f'{col}')
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
        ax.set_xticks(range(0, 25, 1))
        if idx == 19:
            ax.set_xlabel('시간')
        if idx == 0:
            ax.set_ylabel('충방전량 (kW)')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle('EV별 20개 에피소드 그룹별 충방전 패턴(평균)')
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(png_dir, 'smp_and_charging.png'), bbox_inches='tight')
    plt.close()

    # 4. SOC 변화 (EV별 subplot, 그룹별 평균)
    soc_columns = [col for col in df.columns if 'soc' in col and 'ev' in col]
    fig, axes = plt.subplots(20, 1, figsize=(20, 2.5*20), sharex=True, sharey=True)
    for idx, col in enumerate(soc_columns):
        ax = axes[idx]
        grouped_soc = df.groupby(['episode_group', 'time'], as_index=False)[col].mean()
        for group in grouped_soc['episode_group'].unique():
            group_data = grouped_soc[grouped_soc['episode_group'] == group]
            ax.plot(group_data['time'], group_data[col], label=f'Group {int(group)*20}~{int(group)*20+19}')
        ax.set_title(f'{col}')
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
        ax.set_xticks(range(0, 25, 1))
        if idx == 19:
            ax.set_xlabel('시간')
        if idx == 0:
            ax.set_ylabel('SOC (%)')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle('EV별 20개 에피소드 그룹별 SOC 변화(평균)')
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
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