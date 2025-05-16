# V2G 전기차 충방전 스케줄링 DQN

이 프로젝트는 Deep Q-Network(DQN)를 사용하여 Vehicle-to-Grid(V2G) 시스템에서 전기차의 최적 충방전 스케줄을 학습하는 강화학습 모델을 구현합니다.

## 프로젝트 구조

- `environment.py`: V2G 환경 시뮬레이션
- `dqn_agent.py`: DQN 에이전트 구현
- `train.py`: 학습 실행 스크립트
- `requirements.txt`: 필요한 패키지 목록

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

학습 시작:
```bash
python train.py
```

## 환경 설정

- 배터리 용량: 100 kWh
- 최대 충전율: 10 kW
- 최대 방전율: 10 kW
- 시뮬레이션 기간: 24시간
- 전기 가격: 시간대별 변동 (피크/비피크)

## 주요 기능

- 시간대별 전기 가격 변동 고려
- 배터리 상태 관리
- 충방전 최적화를 통한 비용 절감
- 실시간 학습 진행 상황 모니터링
- 학습 결과 시각화

## 결과

학습이 완료되면 `learning_progress.png` 파일에서 학습 진행 상황을 확인할 수 있습니다. 