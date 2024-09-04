import carla

def get_carla_version():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # 서버 연결 시간 제한 설정
    print("CARLA 서버 버전:", client.get_server_version())
    print("CARLA Python API 버전:", client.get_client_version())

get_carla_version()
