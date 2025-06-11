# -*- coding: utf-8 -*-
import numpy as np
import gym
from gym import spaces
from visualization import V2GVisualization
from grid_components import GridComponents

class V2GEnvironment(gym.Env):
    def __init__(self, num_vehicles=20):
        super(V2GEnvironment, self).__init__()
        
        # 차량 수 설정
        self.num_vehicles = num_vehicles
        
        # 그리드 컴포넌트 초기화
        self.grid = GridComponents()
        
        # 환경 파라미터 설정
        self.max_battery_capacity = 100.0  # kWh
        self.max_charging_rate = 7.0      # kW
        self.max_discharging_rate = 10.0   # kW
        self.time_step = 1                 # 1시간 단위
        self.num_time_steps = 24           # 24시간
        self.peak_threshold = 400          # kW - 피크 임계값
        
        # 전기 가격 범위 설정
        self.min_electricity_price = 50    # 원/kWh
        self.max_electricity_price = 200   # 원/kWh
        
        # Action space: 각 차량별 -1(방전) ~ 1(충전) 사이의 연속적인 값
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_vehicles,),
            dtype=np.float32
        )
        
        # State space: [각 차량의 상태(4) * 차량수 + 전기가격 + 건물부하 + 시간]
        self.observation_space = spaces.Box(
            low=np.array([0] * (self.num_vehicles * 4) + [0, 0, 0]),
            high=np.array([100] * (self.num_vehicles * 4) + [1000, 2000, 24]),
            dtype=np.float32
        )
        
        # 각 차량의 상태를 저장하는 딕셔너리
        self.vehicles = {i: {
            'present': False,
            'soc': 0.0,
            'initial_soc': 0.0,
            'target_soc': 0.0,
            'arrival_time': 0,
            'departure_time': 0,
            'charging_rate': 0.0
        } for i in range(self.num_vehicles)}
        
        # 시각화 객체 생성
        self.visualization = V2GVisualization(num_vehicles=self.num_vehicles)
        
        # 현재 시간과 가격 초기화
        self.current_time = 0
        self.current_price = self.grid.get_current_smp()
        self.current_load = self.grid.get_current_building_load()
    
    def _clip_action(self, action, current_soc):
        """안전한 충/방전 범위로 action을 제한"""
        if action > 0:  # 충전
            # 남은 충전 가능 용량 계산
            max_possible_charge = (self.max_battery_capacity - current_soc) / self.max_charging_rate
            return min(action, max_possible_charge)
        else:  # 방전
            # 방전 가능 용량 계산
            max_possible_discharge = current_soc / self.max_discharging_rate
            return max(action, -max_possible_discharge)
    
    def reset(self):
        # 초기 상태 설정
        self.current_time = 0
        self.current_price = self.grid.get_current_smp()
        self.current_load = self.grid.get_current_building_load()
        
        # 각 차량 상태 초기화
        for i in range(self.num_vehicles):
            # 입차 시 SOC는 20~60% 사이로 설정
            initial_soc = np.random.uniform(20, 60)
            
            # 목표 SOC는 입차 SOC보다 최소 20% 높게 설정 (최대 100%)
            min_target = min(initial_soc + 20, 100)  # 최소 20% 증가
            max_target = min(initial_soc + 40, 100)  # 최대 40% 증가
            target_soc = np.random.uniform(min_target, max_target)
            
            arrival_time = np.random.randint(0, 12)
            departure_time = np.random.randint(12, 24)

            # 차량은 도착 시간까지 주차장에 존재하지 않음
            self.vehicles[i]['present'] = arrival_time == 0
            self.vehicles[i]['soc'] = initial_soc
            self.vehicles[i]['initial_soc'] = initial_soc
            self.vehicles[i]['target_soc'] = target_soc
            self.vehicles[i]['arrival_time'] = arrival_time
            self.vehicles[i]['departure_time'] = departure_time
            self.vehicles[i]['charging_rate'] = 0.0
        
        return self._get_observation()
    
    def _get_observation(self):
        obs = []
        for i in range(self.num_vehicles):
            if self.vehicles[i]['present']:
                obs.extend([
                    self.vehicles[i]['soc'] / 100.0,
                    self.vehicles[i]['target_soc'] / 100.0,
                    self.vehicles[i]['charging_rate'] / self.max_charging_rate,
                    self.vehicles[i]['departure_time'] / 24.0
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # 전기 가격, 건물 부하, 시간 정보 추가
        obs.append(self.current_price / 1000.0)  # 정규화
        obs.append(self.current_load / 2000.0)   # 정규화
        obs.append(self.current_time / 24.0)
        
        return np.array(obs)
    
    def step(self, actions):
        total_reward = 0
        total_ev_power = 0
        individual_powers = []  # 각 EV의 전력량 저장

        # 현재 시간의 건물 부하는 스텝 동안 고정
        building_load = self.current_load

        # 1단계: 각 차량의 충/방전량 계산
        for i, action in enumerate(actions):
            if not self.vehicles[i]['present']:
                individual_powers.append(0)
                continue
            
            # 현재 시간이 출발 시간이면 차량 제거
            if self.current_time == self.vehicles[i]['departure_time']:
                self.vehicles[i]['present'] = False
                if self.vehicles[i]['soc'] < self.vehicles[i]['target_soc']:
                    penalty = (self.vehicles[i]['target_soc'] - self.vehicles[i]['soc']) * 10
                    total_reward -= penalty
                continue
            
            # 현재 시간이 도착 시간이면 차량 추가
            if self.current_time == self.vehicles[i]['arrival_time']:
                self.vehicles[i]['present'] = True
            
            # 1. SOC 긴급도 체크
            time_to_departure = self.vehicles[i]['departure_time'] - self.current_time
            soc_gap = self.vehicles[i]['target_soc'] - self.vehicles[i]['soc']
            is_urgent = time_to_departure <= 3 and soc_gap > 0
            
            # 2. 피크 상황 체크 - 건물 부하는 스텝 동안 변경되지 않음
            is_peak = building_load > self.peak_threshold
            
            # 3. 가격 상황 체크
            is_price_high = self.current_price > self.grid.get_average_price()
            
            # 행동 우선순위 결정
            if is_urgent:
                # SOC 충전 우선
                action = max(0, action)  # 강제 충전
            elif is_peak:
                # 피크 저감 우선
                action = min(0, action)  # 강제 방전
            elif is_price_high:
                # 가격 차익 추구
                action = min(0, action)  # 방전 유도
            
            # 안전한 충방전 범위로 action 제한
            current_soc = self.vehicles[i]['soc']
            safe_action = self._clip_action(action, current_soc)
            
            # 충방전량 계산 및 적용
            if safe_action > 0:  # 충전
                power = safe_action * self.max_charging_rate
                cost = power * self.current_price
                self.vehicles[i]['soc'] = min(current_soc + power, self.max_battery_capacity)
                total_reward -= cost
                total_ev_power += power
                individual_powers.append(power)
            else:  # 방전
                power = -safe_action * self.max_discharging_rate
                revenue = power * self.current_price
                self.vehicles[i]['soc'] = max(current_soc - power, 0)
                total_reward += revenue
                total_ev_power -= power
                individual_powers.append(-power)
            
            self.vehicles[i]['charging_rate'] = safe_action * self.max_charging_rate
        
        # 2단계: 피크 저감 상황 확인
        is_peak_time = building_load > self.peak_threshold
        
        # 3단계: 각 EV별 피크 저감 보상 계산
        if is_peak_time:
            total_discharge = sum(p for p in individual_powers if p < 0)
            if total_discharge < 0:  # 방전이 있는 경우
                for i, power in enumerate(individual_powers):
                    if power < 0:  # 방전 중인 EV
                        # 개별 EV의 피크 저감 기여도 계산
                        contribution_ratio = abs(power) / abs(total_discharge)
                        peak_reduction = min(abs(total_discharge), building_load - self.peak_threshold)
                        individual_reward = peak_reduction * contribution_ratio * 5 * self.current_price
                        total_reward += individual_reward
                    elif power > 0:  # 충전 중인 EV
                        # 피크 시간대 충전 페널티
                        penalty = power * 3 * self.current_price
                        total_reward -= penalty
        
        # 다음 시간스텝으로 이동
        self.current_time += 1
        done = self.current_time >= self.num_time_steps
        
        # 그리드 상태 업데이트
        self.grid.step()
        self.current_price = self.grid.get_current_smp()
        self.current_load = self.grid.get_current_building_load()
        
        return self._get_observation(), total_reward, done, {} 

    def calculate_peak_contribution(self, total_ev_power):
        building_load = self.grid.get_current_building_load()
        total_load = building_load + total_ev_power
        
        if building_load > self.peak_threshold and total_ev_power < 0:
            # 개별 EV의 피크 저감 기여도 계산
            contribution_ratio = abs(total_ev_power) / abs(total_load)
            peak_reduction = min(abs(total_ev_power), building_load - self.peak_threshold)
            individual_contribution = peak_reduction * contribution_ratio
            return individual_contribution * 5

    def calculate_total_reward(self, vehicle, action, power, price):
        reward = 0
        
        # 기본 충방전 비용/수익
        if power > 0:
            reward -= power * price
        else:
            reward += abs(power) * price
        
        # SOC 패널티 (더 민감하게)
        time_to_departure = vehicle['departure_time'] - self.current_time
        soc_gap = vehicle['target_soc'] - vehicle['soc']
        if time_to_departure <= 3:
            reward -= abs(soc_gap) * (20 / time_to_departure)  # 출발이 가까울수록 더 큰 패널티
        
        # 피크 저감 보상 (더 강화)
        if self.grid.get_current_building_load() > self.peak_threshold:
            if power < 0:  # 방전
                reward += abs(power) * price * 7  # 피크 저감 보상 증가
            else:  # 충전
                reward -= power * price * 4  # 피크 시 충전 패널티 증가
        
        return reward

    def predict_future_prices(self):
        # 시간대별 평균 가격 패턴 활용
        future_prices = []
        for t in range(self.current_time, 24):
            base_price = self.grid.base_smp_pattern[t]
            season_weight = self.grid.season_weights[self.current_season]['smp']
            day_weight = self.grid.day_weights['weekend' if self.is_weekend else 'weekday']['smp']
            predicted_price = base_price * season_weight * day_weight
            future_prices.append(predicted_price)
        return future_prices 

    def optimize_charging_strategy(self, vehicle, future_prices):
        remaining_time = vehicle['departure_time'] - self.current_time
        required_charge = vehicle['target_soc'] - vehicle['soc']
        
        if required_charge > 0:
            # 남은 시간 동안의 최저가 시간대 찾기
            best_charging_times = sorted(range(len(future_prices)), 
                                       key=lambda i: future_prices[i])[:int(required_charge/self.max_charging_rate)]
            return best_charging_times
        return [] 