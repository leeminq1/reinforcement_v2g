# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d

class GridComponents:
    def __init__(self):
        # SMP 가격 패턴 (원/kWh) - 시간별 기본 패턴
        self.base_smp_pattern = {
            0: 50,   1: 30,   2: 10,   3: -10,  4: -20,  5: 0,
            6: 20,   7: 80,   8: 150,  9: 250,  10: 300, 11: 280,
            12: 200, 13: 180, 14: 190, 15: 220, 16: 280, 17: 350,
            18: 400, 19: 300, 20: 200, 21: 100, 22: 80,  23: 60
        }
        
        # 건물 부하 패턴 (kW) - 시간별 기본 패턴
        self.base_building_load = {
            0: 120,  1: 100,  2: 90,   3: 85,   4: 80,   5: 85,
            6: 100,  7: 150,  8: 200,  9: 250,  10: 280, 11: 300,
            12: 320, 13: 330, 14: 340, 15: 350, 16: 360, 17: 380,
            18: 400, 19: 350, 20: 300, 21: 250, 22: 180, 23: 150
        }
        
        # 계절성 가중치 수정
        self.season_weights = {
            'spring': {
                'smp': 0.7,    # 봄철 더 낮게
                'load': 0.8
            },
            'summer': {
                'smp': 1.8,    # 여름철 더 높게
                'load': 1.5
            },
            'fall': {
                'smp': 0.6,    # 가을철 더 낮게
                'load': 0.7
            },
            'winter': {
                'smp': 1.6,    # 겨울철 더 높게
                'load': 1.3
            }
        }
        
        # 요일 가중치 수정
        self.day_weights = {
            'weekday': {
                'smp': 1.3,    # 주중 더 높게
                'load': 1.2
            },
            'weekend': {
                'smp': 0.5,    # 주말 더 낮게
                'load': 0.6
            }
        }
        
        # 현재 시간
        self.current_hour = 0
        self.current_season = 'summer'  # 초기값
        self.is_weekend = False
        
        # 피크 관련 변수
        self.peak_threshold = 400  # kW - 피크 임계값도 조정
        self.daily_peak = 0
        self.monthly_peak = 0
        
        # 보간 함수 생성
        self._create_interpolation_functions()
    
    def _create_interpolation_functions(self):
        """시간별 패턴을 연속적인 값으로 보간하는 함수 생성"""
        hours = list(self.base_smp_pattern.keys())
        smp_values = list(self.base_smp_pattern.values())
        load_values = list(self.base_building_load.values())
        
        # 24시간을 순환하는 보간을 위해 데이터 확장
        hours_ext = hours + [24]
        smp_values_ext = smp_values + [smp_values[0]]
        load_values_ext = load_values + [load_values[0]]
        
        self.smp_interpolator = interp1d(hours_ext, smp_values_ext, kind='cubic')
        self.load_interpolator = interp1d(hours_ext, load_values_ext, kind='cubic')
    
    def get_average_price(self):
        """현재 시간대의 평균 SMP 가격 계산"""
        base_price = self.base_smp_pattern[self.current_hour % 24]
        season_weight = self.season_weights[self.current_season]['smp']
        day_weight = self.day_weights['weekend' if self.is_weekend else 'weekday']['smp']
        return base_price * season_weight * day_weight
    
    def get_current_smp(self):
        """현재 시간의 SMP 가격 반환"""
        base_price = float(self.smp_interpolator(self.current_hour % 24))
        season_weight = self.season_weights[self.current_season]['smp']
        day_weight = self.day_weights['weekend' if self.is_weekend else 'weekday']['smp']
        
        # 랜덤 변동성 추가 (±30%)
        random_factor = 1 + np.random.uniform(-0.3, 0.3)
        
        # 추가 변동성: 5% 확률로 매우 큰 변동 (-50 ~ 500원/kWh)
        if np.random.random() < 0.05:
            return np.random.uniform(-50, 500)
        
        return base_price * season_weight * day_weight * random_factor
    
    def get_current_building_load(self):
        """현재 시간의 건물 부하 반환"""
        base_load = float(self.load_interpolator(self.current_hour % 24))
        season_weight = self.season_weights[self.current_season]['load']
        day_weight = self.day_weights['weekend' if self.is_weekend else 'weekday']['load']
        
        # 랜덤 변동성 추가 (±10%)
        random_factor = 1 + np.random.uniform(-0.1, 0.1)
        
        current_load = base_load * season_weight * day_weight * random_factor
        
        # 피크 업데이트
        self.daily_peak = max(self.daily_peak, current_load)
        self.monthly_peak = max(self.monthly_peak, current_load)
        
        return current_load
    
    def calculate_peak_contribution(self, ev_power):
        """EV의 피크 저감 기여도 계산"""
        building_load = self.get_current_building_load()
        total_load = building_load + ev_power
        
        # 피크 저감 보상 계산
        if building_load > self.peak_threshold:
            if ev_power < 0:  # 방전(Grid로 전력 공급)
                # 피크 저감에 기여하는 경우
                peak_contribution = min(-ev_power, building_load - self.peak_threshold)
                return peak_contribution * 5  # 피크 저감 보상 5배로 증가
            else:  # 충전(Grid에서 전력 소비)
                # 피크 시간대에 충전하는 경우 페널티
                return -abs(ev_power) * 3  # 피크 시간대 충전 페널티 3배
        elif total_load > self.peak_threshold:
            # EV 충전으로 인해 피크 발생시 큰 페널티
            return -(total_load - self.peak_threshold) * 4  # 피크 유발 페널티 4배
        
        return 0
    
    def step(self):
        """시간 진행"""
        self.current_hour += 1
        if self.current_hour % 24 == 0:
            self.daily_peak = 0  # 일일 피크 초기화
            # 요일 변경
            if self.current_hour % (24 * 7) == 0:
                self.is_weekend = not self.is_weekend
        
        # 월간 피크 초기화 (30일 기준)
        if self.current_hour % (24 * 30) == 0:
            self.monthly_peak = 0
            # 계절 변경
            seasons = ['spring', 'summer', 'fall', 'winter']
            current_season_idx = seasons.index(self.current_season)
            self.current_season = seasons[(current_season_idx + 1) % 4] 