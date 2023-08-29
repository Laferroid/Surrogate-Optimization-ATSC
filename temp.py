# region 错误做法
# while True: 
#     # 假设到达车流由周围交叉口的三个进口道平均贡献
#     headway = 3600/(1/3*vph_level[inlet]*turn_ratio[inlet,turn])
#     headway = np.random.exponential(headway)
#     if (second-headway)>0:
#         second -= headway
#         n_veh += 1
#     else:
#         break
# endregion