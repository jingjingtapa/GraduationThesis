import carla

# 클라이언트 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Town04 맵 로드
world = client.load_world('Town04')

# Town04 맵이 성공적으로 로드되었는지 확인
print(f"현재 로드된 맵: {world.get_map().name}")

