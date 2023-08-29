# import xml.etree.ElementTree as et
# from xml.dom import minidom
import numpy as np
import traci
import time

# 映射到编号
inlet_map = {"N": 0, "E": 1, "S": 2, "W": 3}
direction_map = {"L": 0, "T": 1, "R": 2}

scheme2func = [[2, 2], [2, 1], [2, 0], [1, 0]]

# 根据车辆的路径id获取信息
def get_movement(object_id):
    # object_id为vehicle_id或route_id
    # 示例：route_id='WS_N' 西进口北出口
    # movement为0表示该网格无车
    o = inlet_map[object_id[0]]
    d = inlet_map[object_id[3]]
    # 返回值：[进口道编号，转向编号，出口道编号]
    return (o, (d - o) % 4 - 1, d)


def try_connect(num, sumocfg):
    for _ in range(num):
        try:
            traci.start(sumocfg)
            break
        except Exception:
            time.sleep(0.5)


def sumo_configurate(config):
    sumocfg = [
        "sumo",
        "--route-files",
        "test.rou.xml",
        "--net-file",
        "test.net.xml",
        "--additional-files",
        "test.e2.xml,test.e3.xml",
        "--gui-settings-file",
        "gui.cfg",
        "--delay",
        "0",
        "--time-to-teleport",
        "600",
        "--step-length",
        f"{config['step_length']}",
        "--no-step-log",
        "true",
        "-X",
        "never",
        "--quit-on-end",
        "--save-state.rng",
    ]
    return sumocfg


# <控制格式>: 包括phase key和split key的字典，value为array格式，split为严格满足约束的整数变量，phase为0-1变量
