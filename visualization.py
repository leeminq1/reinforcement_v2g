# -*- coding: utf-8 -*-
import pygame
import numpy as np

class V2GVisualization:
    def __init__(self, num_vehicles=20):
        pygame.init()
        
        # Screen setup
        self.width = 1400
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("V2G Simulation")
        
        # 상수 정의
        self.max_electricity_price = 500  # 원/kWh
        self.max_building_load = 2000     # kW
        self.peak_threshold = 400         # kW
        
        # 히스토리 관련 변수 초기화
        self.max_price_history = 48
        self.price_history = []
        self.original_load_history = []   # 원래 건물 부하
        self.controlled_load_history = [] # EV 충방전이 반영된 부하
        self.current_load = 0
        self.current_total_load = 0
        
        # Color definitions
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.LIGHT_BLUE = (200, 200, 255)
        self.LIGHT_GRAY = (240, 240, 240)
        
        # Font setup
        self.font = pygame.font.SysFont('arial', 24)
        self.small_font = pygame.font.SysFont('arial', 16)
        self.graph_font = pygame.font.SysFont('arial', 14)
        
        # Vehicle settings
        self.num_vehicles = num_vehicles
        self.vehicle_width = 75
        self.vehicle_height = 38
        self.spacing_x = 155
        self.spacing_y = 200
        
        # 초기화 추가
        self.current_time = 0
        self.current_price = 0
        self.total_reward = 0
        self.episode = 0
        
        # 주차 공간 생성
        self.parking_spots = []
        max_vehicles_per_row = 7
        num_rows = (num_vehicles + max_vehicles_per_row - 1) // max_vehicles_per_row
        
        for row in range(num_rows):
            vehicles_in_row = min(max_vehicles_per_row, num_vehicles - row * max_vehicles_per_row)
            row_width = vehicles_in_row * self.spacing_x
            start_x = (self.width - row_width) // 2
            
            for col in range(vehicles_in_row):
                x = start_x + col * self.spacing_x
                y = 220 + row * self.spacing_y
                self.parking_spots.append((x, y))
        
        # Vehicle states
        self.vehicle_states = {i: {
            'present': False,
            'soc': 0.0,
            'initial_soc': 0.0,
            'target_soc': 0.0,
            'charging_rate': 0.0,
            'arrival_time': 0,
            'departure_time': 0
        } for i in range(num_vehicles)}
        
    def draw_info_panel(self):
        # Main info panel - 크기 축소
        panel_rect = pygame.Rect(20, 20, 350, 160)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.BLACK, panel_rect, 2)
        
        # 정보 텍스트 정렬을 위한 여백
        margin_left = 30
        line_height = 35
        
        # Time info
        time_text = f"Time: {self.current_time:02d}:00"
        text_surface = self.font.render(time_text, True, self.BLACK)
        self.screen.blit(text_surface, (margin_left, 30))
        
        # Price info
        price_text = f"Price: {self.current_price:.1f} KRW/kWh"
        text_surface = self.font.render(price_text, True, self.BLACK)
        self.screen.blit(text_surface, (margin_left, 30 + line_height))
        
        # Episode info
        episode_text = f"Episode: {self.episode}"
        text_surface = self.font.render(episode_text, True, self.BLACK)
        self.screen.blit(text_surface, (margin_left, 30 + line_height * 2))
        
        # Reward info
        reward_text = f"Total Reward: {self.total_reward:.1f}"
        text_surface = self.font.render(reward_text, True, self.BLACK)
        self.screen.blit(text_surface, (margin_left, 30 + line_height * 3))
        
        # Draw price history graph
        self.draw_price_history()
        
    def draw_price_history(self):
        # SMP 가격 그래프
        price_graph_rect = pygame.Rect(400, 20, 450, 160)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, price_graph_rect)
        pygame.draw.rect(self.screen, self.BLACK, price_graph_rect, 2)
        
        # 건물 부하 그래프
        load_graph_rect = pygame.Rect(900, 20, 450, 160)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, load_graph_rect)
        pygame.draw.rect(self.screen, self.BLACK, load_graph_rect, 2)
        
        # 그래프 제목
        price_title = self.font.render("SMP Price History", True, self.BLACK)
        price_title_rect = price_title.get_rect(midtop=(price_graph_rect.centerx, 25))
        self.screen.blit(price_title, price_title_rect)
        
        load_title = self.font.render("Building Load History", True, self.BLACK)
        load_title_rect = load_title.get_rect(midtop=(load_graph_rect.centerx, 25))
        self.screen.blit(load_title, load_title_rect)
        
        # 히스토리 업데이트
        self.price_history.append(self.current_price)
        if len(self.price_history) > self.max_price_history:
            self.price_history.pop(0)
            
        self.original_load_history.append(self.current_load)
        if len(self.original_load_history) > self.max_price_history:
            self.original_load_history.pop(0)
            
        self.controlled_load_history.append(self.current_total_load)
        if len(self.controlled_load_history) > self.max_price_history:
            self.controlled_load_history.pop(0)
        
        # 그래프 그리기
        if len(self.price_history) > 1:
            # 가격 그래프
            max_price = max(max(self.price_history), 1)
            min_price = min(min(self.price_history), 0)
            price_range = max_price - min_price + 1
            
            # 가격 그리드 라인과 레이블
            for i in range(4):
                y = price_graph_rect.bottom - 30 - (i * (price_graph_rect.height - 60) / 3)
                pygame.draw.line(self.screen, self.GRAY, 
                               (price_graph_rect.left + 50, y),
                               (price_graph_rect.right - 20, y), 1)
                
                price = min_price + (price_range * i / 3)
                price_label = self.graph_font.render(f"{price:.0f}원", True, self.BLUE)
                price_rect = price_label.get_rect(right=price_graph_rect.left + 45, centery=y)
                self.screen.blit(price_label, price_rect)
            
            # 가격선 그리기
            price_points = []
            for i, price in enumerate(self.price_history):
                x = int(price_graph_rect.left + 50 + (i * (price_graph_rect.width - 70) / self.max_price_history))
                y = int(price_graph_rect.bottom - 30 - ((price - min_price) * (price_graph_rect.height - 60) / price_range))
                price_points.append((x, y))
            
            if len(price_points) >= 2:
                pygame.draw.lines(self.screen, self.BLUE, False, price_points, 2)
            
            # 부하 그래프
            all_loads = self.original_load_history + self.controlled_load_history
            max_load = max(max(all_loads), self.peak_threshold, 1)
            min_load = min(min(all_loads), 0)
            load_range = max_load - min_load + 1
            
            # 부하 그리드 라인과 레이블
            for i in range(4):
                y = load_graph_rect.bottom - 30 - (i * (load_graph_rect.height - 60) / 3)
                pygame.draw.line(self.screen, self.GRAY, 
                               (load_graph_rect.left + 50, y),
                               (load_graph_rect.right - 20, y), 1)
                
                load = min_load + (load_range * i / 3)
                load_label = self.graph_font.render(f"{load:.0f}kW", True, self.RED)
                load_rect = load_label.get_rect(right=load_graph_rect.left + 45, centery=y)
                self.screen.blit(load_label, load_rect)
            
            # 원래 부하선 그리기 (빨간색)
            original_points = []
            for i, load in enumerate(self.original_load_history):
                x = int(load_graph_rect.left + 50 + (i * (load_graph_rect.width - 70) / self.max_price_history))
                y = int(load_graph_rect.bottom - 30 - ((load - min_load) * (load_graph_rect.height - 60) / load_range))
                original_points.append((x, y))
            
            if len(original_points) >= 2:
                pygame.draw.lines(self.screen, self.RED, False, original_points, 2)
            
            # 제어된 부하선 그리기 (주황색)
            controlled_points = []
            for i, load in enumerate(self.controlled_load_history):
                x = int(load_graph_rect.left + 50 + (i * (load_graph_rect.width - 70) / self.max_price_history))
                y = int(load_graph_rect.bottom - 30 - ((load - min_load) * (load_graph_rect.height - 60) / load_range))
                controlled_points.append((x, y))
            
            if len(controlled_points) >= 2:
                pygame.draw.lines(self.screen, (255, 165, 0), False, controlled_points, 2)
            
            # 피크 임계선 그리기 (회색 점선)
            peak_y = int(load_graph_rect.bottom - 30 - ((self.peak_threshold - min_load) * (load_graph_rect.height - 60) / load_range))
            for x in range(load_graph_rect.left + 50, load_graph_rect.right - 20, 10):
                pygame.draw.line(self.screen, self.DARK_GRAY,
                               (x, peak_y),
                               (x + 5, peak_y), 2)
            
            # 범례 추가
            legend_y = load_graph_rect.bottom - 20
            # 원래 부하 범례
            pygame.draw.line(self.screen, self.RED, 
                           (load_graph_rect.left + 60, legend_y),
                           (load_graph_rect.left + 90, legend_y), 2)
            original_text = self.graph_font.render("Original Load", True, self.BLACK)
            self.screen.blit(original_text, (load_graph_rect.left + 100, legend_y - 7))
            
            # 제어된 부하 범례
            pygame.draw.line(self.screen, (255, 165, 0),
                           (load_graph_rect.left + 200, legend_y),
                           (load_graph_rect.left + 230, legend_y), 2)
            controlled_text = self.graph_font.render("Controlled Load", True, self.BLACK)
            self.screen.blit(controlled_text, (load_graph_rect.left + 240, legend_y - 7))
            
            # X축 시간 레이블
            for graph_rect in [price_graph_rect, load_graph_rect]:
                for i in range(0, self.max_price_history, 6):
                    x = graph_rect.left + 50 + (i * (graph_rect.width - 70) / self.max_price_history)
                    time = (self.current_time - (self.max_price_history - i)) % 24
                    if time < 0:
                        time += 24
                    label = self.graph_font.render(f"{time:02d}:00", True, self.BLACK)
                    label_rect = label.get_rect(midtop=(x, graph_rect.bottom - 25))
                    self.screen.blit(label, label_rect)
        
    def draw_vehicle(self, pos, soc, charging_rate, index):
        x, y = map(int, pos)
        
        # SOC bar (차량 왼쪽에 배치)
        soc_height = int(self.vehicle_height * 1.5)
        soc_width = 8
        soc_x = x - 15
        soc_y = y - 8
        
        # SOC bar background
        soc_bg_rect = pygame.Rect(soc_x, soc_y, soc_width, soc_height)
        pygame.draw.rect(self.screen, self.BLACK, soc_bg_rect)
        
        # Fill SOC bar
        soc_fill_height = int(soc_height * max(0, min(soc, 100)) / 100)
        if soc_fill_height > 0:
            soc_fill_rect = pygame.Rect(
                soc_x,
                soc_y + (soc_height - soc_fill_height),
                soc_width,
                soc_fill_height
            )
            fill_color = self.BLUE
            pygame.draw.rect(self.screen, fill_color, soc_fill_rect)
        
        # 차량 색상 결정 (피크 저감 시 주황색)
        if charging_rate < 0 and self.current_load > self.peak_threshold:
            car_color = (255, 165, 0)  # 주황색 (피크 저감)
        else:
            car_color = self.GREEN if charging_rate > 0 else (self.RED if charging_rate < 0 else self.GRAY)
        
        # Draw car
        self.draw_car(x, y, car_color)
        
        # Vehicle info background
        info_bg = pygame.Rect(x - 15, y + self.vehicle_height + 8, 120, 85)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, info_bg)
        pygame.draw.rect(self.screen, self.BLACK, info_bg, 1)
        
        # Info text positioning
        text_x = x - 10
        text_y = y + self.vehicle_height + 10
        line_spacing = 16
        
        # Vehicle info text
        texts = [
            (f"EV{index} - SOC: {soc:.1f}%", self.BLACK),
            (f"Initial: {self.vehicle_states[index]['initial_soc']:.1f}%", self.GRAY),
            (f"Target: {self.vehicle_states[index]['target_soc']:.1f}%", self.RED),
            (f"Rate: {charging_rate:.1f}kW", self.BLACK)
        ]
        
        for i, (text, color) in enumerate(texts):
            text_surface = self.small_font.render(text, True, color)
            self.screen.blit(text_surface, (text_x, text_y + i * line_spacing))
        
        # 입/출차 시간
        if self.vehicle_states[index]['present']:
            arrival = self.vehicle_states[index]['arrival_time']
            departure = self.vehicle_states[index]['departure_time']
            time_text = f"In: {arrival:02d}:00 Out: {departure:02d}:00"
            time_surface = self.small_font.render(time_text, True, self.BLACK)
            self.screen.blit(time_surface, (text_x, text_y + 4 * line_spacing))
        
        # Draw charging arrow
        if abs(charging_rate) > 0:
            arrow_color = self.GREEN if charging_rate > 0 else self.RED
            arrow_x = x + self.vehicle_width // 2
            if charging_rate > 0:
                arrow_start = (arrow_x, y - 15)
                arrow_end = (arrow_x, y - 5)
                pygame.draw.polygon(self.screen, arrow_color, [
                    (arrow_x - 8, y - 15),
                    (arrow_x + 8, y - 15),
                    (arrow_x, y - 23)
                ])
            else:
                arrow_start = (arrow_x, y - 5)
                arrow_end = (arrow_x, y - 15)
                pygame.draw.polygon(self.screen, arrow_color, [
                    (arrow_x - 8, y - 23),
                    (arrow_x + 8, y - 23),
                    (arrow_x, y - 15)
                ])
            pygame.draw.line(self.screen, arrow_color, arrow_start, arrow_end, 3)

    def draw_car(self, x, y, color):
        # Car body
        pygame.draw.rect(self.screen, color, (x, y + 15, self.vehicle_width, self.vehicle_height - 15))
        
        # Roof
        roof_points = [
            (x + 15, y + 15),
            (x + 30, y),
            (x + self.vehicle_width - 30, y),
            (x + self.vehicle_width - 15, y + 15)
        ]
        pygame.draw.polygon(self.screen, color, roof_points)
        
        # Wheels
        wheel_radius = 8
        wheel_positions = [
            (x + 25, y + self.vehicle_height - 3),
            (x + self.vehicle_width - 25, y + self.vehicle_height - 3)
        ]
        for wx, wy in wheel_positions:
            pygame.draw.circle(self.screen, self.BLACK, (wx, wy), wheel_radius)
            pygame.draw.circle(self.screen, self.DARK_GRAY, (wx, wy), wheel_radius - 3)
            
    def update(self, vehicle_states, current_time, current_price, total_reward, episode, current_load):
        self.vehicle_states = vehicle_states
        self.current_time = current_time
        self.current_price = current_price
        self.total_reward = total_reward
        self.episode = episode
        self.current_load = current_load
        
        # EV 충방전 전력 계산
        total_ev_power = sum([self.vehicle_states[i]['charging_rate'] for i in range(self.num_vehicles) if self.vehicle_states[i]['present']])
        self.current_total_load = current_load + total_ev_power
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw info panel
        self.draw_info_panel()
        
        # Draw vehicles
        for i, pos in enumerate(self.parking_spots):
            if self.vehicle_states[i]['present']:
                self.draw_vehicle(pos, 
                                self.vehicle_states[i]['soc'],
                                self.vehicle_states[i]['charging_rate'],
                                i)
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True
    
    def close(self):
        pygame.quit() 