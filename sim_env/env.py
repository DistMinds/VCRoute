from sim_env.wall_time import WallTime
from sim_env.timeline import Timeline
from sim_env.task import SendTask, ReceiveTask
import time
from sim_env.packet import Packet, Window
from param import *
import numpy as np
import pandas as pd
import os
from _collections import defaultdict
from sim_env.jitter import Jitter
import networkx as nx
from sim_env.ucbvia import SlotMachine
import matplotlib.pyplot as plt
import random


def plot_prediction(test_result, predict_result):
    plt.plot(test_result, color='red', label='test')
    plt.plot(predict_result, color='blue', label='predict')
    plt.title('End-to-end latency')
    plt.legend()
    plt.show()


class Environment(object):

    def __init__(self):
        # system state
        self.log = None
        self.node_num = 3
        self.usr_num = 3
        self.usrs = []
        self.dcs = []
        self.usr_loc = [0, 1, 2]
        self.total_time = 50000
        self.lt = np.array([[[0, 20, 30], [20, 0, 30], [30, 20, 0]]] * self.total_time)
        self.process_lt = 50
        self.ptnum_per_rt = 50
        self.stream = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        self.stream_num = len(self.stream)
        self.stream_packet_interval = [10] * self.stream_num
        self.packet_num = [0] * self.stream_num
        self.received_packet_num = [0] * self.stream_num
        self.sended_packet_num = [0] * self.stream_num

        self.stream_wan_lt = [[] for i in range(self.stream_num)]
        self.stream_prc_lt = [[] for i in range(self.stream_num)]
        self.stream_lt = [[] for i in range(self.stream_num)]
        self.stream_loss_rate = [0] * self.stream_num
        self.stream_loss_count = [0] * self.stream_num
        self.stream_loss_count_last_period = [0] * self.stream_num
        self.stream_loss_rates = [[] for i in range(self.stream_num)]
        self.stream_output = [[] for i in range(self.stream_num)]
        self.stream_loss_id = [[] for i in range(self.stream_num)]

        # buffer state
        self.buffer = [[] for i in range(self.stream_num)]
        # the idx of max packets in the buffer
        self.buffer_size = [10] * self.stream_num
        self.buffer_free_size = [10] * self.stream_num
        self.crt_output_timestamp = [0] * self.stream_num

        # watermark state
        self.watermark = [0] * self.stream_num
        self.window_size = [5 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_size_new = [5 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_size_max = [100 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_size_min = [1 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_period = [100 for i in range(self.stream_num)]
        self.window_size_ddl = [0 for i in range(self.stream_num)]
        self.loss_rate_budget = 0.05
        self.crr_packet_window = [[0, 5 * self.stream_packet_interval[i]] for i in range(self.stream_num)]
        self.latest_closed_window = [[0, 0] for i in range(self.stream_num)]
        self.latest_open_window = [[0, 0] for i in range(self.stream_num)]
        self.windows = [defaultdict(Window) for i in range(self.stream_num)]
        self.late_tolerant = [0] * self.stream_num
        self.last_delay = [0] * self.stream_num
        self.last_timestamp = [0] * self.stream_num
        self.newest_timestamp = [0] * self.stream_num
        self.jitter = [[] for i in range(self.stream_num)]
        self.late_tolerant_probability = [95] * self.stream_num
        # isolated random idx generator
        self.np_random = np.random.RandomState()

        # global timer
        self.wall_time = WallTime()

        # uses priority queue
        self.timeline = Timeline()

        self.process_algorithm = 'watermark'
        self.route_alg = 'direct'
        self.specified_route = None
        self.generate_send_tasks()
        self.webRTC_jitter = Jitter()

        # Via
        self.via_optobj = 'latency'
        self.via_obj_matrix = np.zeros(self.lt.shape)
        self.via_ltnormals = [0] * self.stream_num
        self.via_toppaths = [None] * self.stream_num
        self.via_all_pulls = [[]] * self.stream_num
        self.via_sms = [[] for i in range(self.stream_num)]
        self.via_prelts = [[] for i in range(self.stream_num)]

        # vcroute
        self.rfvia_obj_matrix = np.zeros(self.lt.shape)
        self.rf_jittermt = np.zeros(self.lt.shape)
        self.rfvia_ltnormals = [0] * self.stream_num
        self.rfvia_toppaths = [None] * self.stream_num
        self.rfvia_sms = [[] for i in range(self.stream_num)]
        self.rfvia_all_pulls = [[0] * self.packet_num[i] for i in range(self.stream_num)]
        self.rfvia_prelts = [[] for i in range(self.stream_num)]
        self.rfvia_prejts = [[] for i in range(self.stream_num)]
        self.rfvia_path_jitter = None
        self.rfvia_predict_jitter = [Jitter() for i in range(self.stream_num)]
        self.rfvia_model = None
        self.rf_output = [[0] * self.packet_num[i] for i in range(self.stream_num)]
        self.rf_lag = 20
        self.rf_n_ahead = 20
        self.rf_n_ft = 0

        self.tmp_matrix = np.zeros((1, self.node_num, self.node_num))

    def step(self, route_begin, route_end, time_start, time_end, total_time, route_alg, process_alg):
        pathname = r'.\result'
        name = str(route_begin) + str(route_end) + str(time_start) + str(time_end)
        self.log = open(os.path.join(pathname, name + 'log.txt'), mode='w')
        self.alterRoute(route_begin, route_end)
        self.reset(time_start, time_end, total_time, route_alg, process_alg)
        crr_routes = [[] for i in range(self.stream_num)]
        crr_route_ids = [0] * self.stream_num
        route_ids = [[] for i in range(self.stream_num)]
        etelts = [[] for i in range(self.stream_num)]
        # 1、数据包生成事件已经压入timeline
        while self.timeline.__len__() != 0:
            event, new_time = self.timeline.pop_task()
            self.wall_time.update_time(new_time)

            # 事件1 发出一个数据包，触发数据包到达事件，更新数据包发送时间，压入数据包到达事件，计算数据包网络延迟，
            if isinstance(event, SendTask):
                finished_send_task = event
                packet = finished_send_task.packet
                packet.set_send_time(new_time)
                stream_id = finished_send_task.stream_id
                send_index = int(new_time / self.stream_packet_interval[stream_id])
                self.sended_packet_num[stream_id] += 1
                if packet.idx % self.ptnum_per_rt == 0:
                    route, route_id = self.generate_route(send_index, packet, stream_id)
                    crr_routes[stream_id] = route
                    crr_route_ids[stream_id] = route_id
                else:
                    route = crr_routes[stream_id]
                    route_id = crr_route_ids[stream_id]
                route_ids[stream_id].append(route_id)
                print(send_index, packet.idx, route)
                packet.set_route(route)
                packet.route_id = route_id
                estimate_wan_lt = self.cal_route_lt(route, self.lt[send_index])
                estimate_event_time = new_time + estimate_wan_lt
                new_receive_task = ReceiveTask(finished_send_task.idx, estimate_event_time, stream_id, packet)
                self.timeline.add_task(new_receive_task, estimate_event_time)

            # 事件2 一个数据包到达，触发数据包处理事件，
            elif isinstance(event, ReceiveTask):
                finished_receive_task = event
                stream_id = finished_receive_task.stream_id
                self.received_packet_num[stream_id] += 1
                org, dst = self.stream[stream_id]
                arrival_packet = finished_receive_task.packet
                arrival_packet.set_arrival_time(new_time)
                wan_lt = new_time - arrival_packet.timestamp
                self.stream_wan_lt[stream_id].append(wan_lt)

                if self.process_algorithm == 'buffer':
                    if self.buffer_free_size[stream_id] > 1:
                        self.buffer[stream_id].append(arrival_packet)
                        self.buffer_free_size[stream_id] = self.buffer_free_size[stream_id] - 1
                    elif self.buffer_free_size[stream_id] == 1:
                        self.buffer[stream_id].append(arrival_packet)
                        tic = time.time()
                        self.buffer[stream_id] = sorted(self.buffer[stream_id],
                                                        key=lambda data_packet: data_packet.timestamp)
                        toc = time.time()
                        sort_lt = (toc - tic) * 1000
                        output_time = new_time + sort_lt
                        # 更新stream_prc_lt, stream_lt, stream_loss_rate，清空buffer，更新系统状态当前最近输出packet的时间戳
                        for packet in self.buffer[stream_id]:
                            packet.set_output_time(output_time)
                            prc_lt = output_time - packet.arrival_time
                            lt = output_time - packet.timestamp
                            self.stream_prc_lt[stream_id].append(prc_lt)
                            self.stream_lt[stream_id].append(lt)
                            if packet.timestamp >= self.crt_output_timestamp[stream_id]:
                                self.stream_output[stream_id].append(packet)
                                self.crt_output_timestamp[stream_id] = packet.timestamp
                            else:
                                self.stream_loss_count[stream_id] = self.stream_loss_count[stream_id] + 1
                        self.buffer[stream_id] = []
                        self.buffer_free_size[stream_id] = self.buffer_size[stream_id]
                    else:
                        tic = time.time()
                        self.buffer[stream_id] = sorted(self.buffer[stream_id],
                                                        key=lambda data_packet: data_packet.timestamp)
                        toc = time.time()
                        sort_lt = (toc - tic) * 1000
                        output_time = new_time + sort_lt
                        # 更新stream_prc_lt, stream_lt, stream_loss_rate，清空buffer，更新系统状态当前最近输出packet的时间戳
                        for packet in self.buffer[stream_id]:
                            packet.set_output_time(output_time)
                            prc_lt = output_time - packet.arrival_time
                            lt = output_time - packet.timestamp
                            self.stream_prc_lt[stream_id].append(prc_lt)
                            self.stream_lt[stream_id].append(lt)
                            if packet.timestamp >= self.crt_output_timestamp[stream_id]:
                                self.stream_output[stream_id].append(packet)
                                self.crt_output_timestamp[stream_id] = packet.timestamp
                            else:
                                self.stream_loss_count[stream_id] = self.stream_loss_count[stream_id] + 1
                        self.buffer[stream_id] = []
                        self.buffer[stream_id].append(arrival_packet)
                        self.buffer_free_size[stream_id] = self.buffer_size[stream_id] - 1

                elif self.process_algorithm == 'watermark':
                    relative_delay = self.webRTC_jitter.CalculateTargetLevel(arrival_packet.timestamp,
                                                                             arrival_packet.arrival_time)
                    self.late_tolerant[stream_id] = self.webRTC_jitter.target_level_ms_
                    arrival_packet.send_info[1:6] = [wan_lt, arrival_packet.timestamp,
                                                     0, self.watermark[stream_id], self.late_tolerant[stream_id]]
                    print('watermark', self.watermark[stream_id], 'lag', self.late_tolerant[stream_id],
                          'arrival packet ts', arrival_packet.timestamp)
                    # 是否丢弃该包
                    if arrival_packet.timestamp < self.watermark[stream_id]:
                        self.stream_loss_id[stream_id].append(arrival_packet.idx)
                        self.stream_loss_count[stream_id] = self.stream_loss_count[stream_id] + 1
                    else:
                        self.buffer[stream_id].append(arrival_packet)
                    # 更新watermark
                    if self.received_packet_num[stream_id] == 1:
                        self.watermark[stream_id] = -1
                    else:
                        if self.last_timestamp[stream_id] > self.watermark[stream_id] + self.late_tolerant[stream_id] - \
                                self.stream_packet_interval[stream_id]:
                            self.watermark[stream_id] = self.last_timestamp[stream_id] + self.stream_packet_interval[
                                stream_id] - self.late_tolerant[stream_id]
                            self.windows[stream_id][0] = Window(stream_id, self.window_size[stream_id], 0, 0, [])
                            for i in range(len(self.buffer[stream_id]) - 1, -1, -1):
                                packet = self.buffer[stream_id][i]
                                if self.watermark[stream_id] >= packet.timestamp:
                                    self.buffer[stream_id].pop(i)
                                    self.windows[stream_id][0].append_packet(packet)
                            tic = time.time()
                            self.windows[stream_id][0].sort_packets()
                            toc = time.time()
                            sort_lt = (toc - tic) * 1000
                            output_time = new_time + sort_lt
                            for packet in self.windows[stream_id][0].packets:
                                packet.set_output_time(output_time)
                                prc_lt = output_time - packet.arrival_time
                                lt = output_time - packet.timestamp
                                self.stream_prc_lt[stream_id].append(prc_lt)
                                self.stream_lt[stream_id].append(lt)
                                if self.route_alg == 'vcroute':
                                    self.rf_output[stream_id][packet.idx] = lt
                                print("output packet: ", packet.idx)
                                packet.send_info[6] = lt
                                self.stream_output[stream_id].append(packet)
                            self.windows[stream_id].pop(0)
                    self.last_timestamp[stream_id] = arrival_packet.timestamp
                    # 收到最后一个包，将所有packet输出
                    if self.received_packet_num[stream_id] == self.packet_num[stream_id]:
                        self.windows[stream_id][0] = Window(stream_id, self.window_size[stream_id], 0, 0, [])
                        for i in range(len(self.buffer[stream_id]) - 1, -1, -1):
                            packet = self.buffer[stream_id][i]
                            self.buffer[stream_id].pop(i)
                            self.windows[stream_id][0].append_packet(packet)
                        tic = time.time()
                        self.windows[stream_id][0].sort_packets()
                        toc = time.time()
                        sort_lt = (toc - tic) * 1000
                        output_time = new_time + sort_lt
                        for packet in self.windows[stream_id][0].packets:
                            packet.set_output_time(output_time)
                            prc_lt = output_time - packet.arrival_time
                            lt = output_time - packet.timestamp
                            self.stream_prc_lt[stream_id].append(prc_lt)
                            self.stream_lt[stream_id].append(lt)
                            if self.route_alg == 'vcroute':
                                self.rf_output[stream_id][packet.idx] = lt
                            print("output packet: ", packet.idx)
                            packet.send_info[6] = lt
                            self.stream_output[stream_id].append(packet)
                        self.windows[stream_id].pop(0)

                else:
                    tic = time.time()
                    self.webRTC_jitter.InsertPacketInternal(arrival_packet.idx, arrival_packet.size,
                                                            arrival_packet.timestamp, arrival_packet.arrival_time,
                                                            arrival_packet.route)
                    buffer_size = self.webRTC_jitter.target_level_ms_ / self.webRTC_jitter.kBucketSizeMs
                    toc = time.time()
                    print(arrival_packet.idx, buffer_size, self.webRTC_jitter.number_of_packet)
                    insert_delay = (toc - tic) * 1000
                    self.webRTC_jitter.packet_buffer[
                        self.webRTC_jitter.insert_index].totaldelay += insert_delay + self.process_lt
                    self.webRTC_jitter.packet_buffer[
                        self.webRTC_jitter.insert_index].prc_lt += insert_delay + self.process_lt
                    if self.received_packet_num[stream_id] == self.packet_num[stream_id]:
                        self.webRTC_jitter.Flush(new_time)
                        errorNum = 0
                        last_Number = self.webRTC_jitter.outputData[0].idx
                        for i in range(1, len(self.webRTC_jitter.outputData)):
                            cur_number = self.webRTC_jitter.outputData[i].idx
                            if last_Number > cur_number:
                                self.stream_loss_id[stream_id].append(cur_number)
                                errorNum = errorNum + 1
                                print("wrong id: ", cur_number)
                            else:
                                last_Number = cur_number
                                self.stream_prc_lt[stream_id].append(self.webRTC_jitter.outputData[i].prc_lt)
                                self.stream_lt[stream_id].append(self.webRTC_jitter.outputData[i].totaldelay)
                                self.stream_output[stream_id].append(self.webRTC_jitter.outputData[i])
                        self.stream_loss_count[stream_id] = errorNum

            else:
                print('finished')
                exit(1)
        self.stream_loss_rate = [self.stream_loss_count[i] / self.packet_num[i] for i in range(self.stream_num)]
        self.log.close()
        routeresult = open(os.path.join(pathname, 'route' + name + '.txt'), mode='w')
        print("the route is ", self.stream, "  the begin index and end index is ", time_start, " ", time_end)
        for i, packet in enumerate(self.stream_output[stream_id]):
            routeresult.writelines([str(packet.idx), str(packet.route), str(self.stream_lt[stream_id][i]), '\t',
                                    str(self.stream_prc_lt[stream_id][i]), '\n'])
        routeresult.close()
        np.savetxt(os.path.join(pathname, 'etelt' + name + '.csv'), np.array(
            [self.stream_lt[stream_id][-int(0.8 * self.packet_num[stream_id]):],
             self.stream_prc_lt[stream_id][-int(0.8 * self.packet_num[stream_id]):]]).T, delimiter=",")
        print("process latency is ", np.average(self.stream_prc_lt[stream_id]), "; loss packet idx is ",
              self.stream_loss_count, "; loss rate is ", self.stream_loss_rate,
              "; sum of wan latency and process latency is ",
              np.average(self.stream_wan_lt[stream_id]) + np.average(self.stream_prc_lt[stream_id]),
              "; end to end latency is ",
              np.average(self.stream_lt[stream_id]))
        print(self.stream_loss_id)
        print('last 10% packet, process latency, end to end latency')
        print(np.average(self.stream_prc_lt[stream_id][-int(0.1 * self.packet_num[stream_id]):]),
              np.average(self.stream_lt[stream_id][-int(0.1 * self.packet_num[stream_id]):]))
        print('last 50% packet, process latency, end to end latency')
        print(np.average(self.stream_prc_lt[stream_id][-int(0.5 * self.packet_num[stream_id]):]),
              np.average(self.stream_lt[stream_id][-int(0.5 * self.packet_num[stream_id]):]))
        print('last 80% packet, process latency, end to end latency')
        print(np.average(self.stream_prc_lt[stream_id][-int(0.8 * self.packet_num[stream_id]):]),
              np.average(self.stream_lt[stream_id][-int(0.8 * self.packet_num[stream_id]):]))

        return self.timeline.__len__() == 0

    def generate_send_tasks(self):
        for i in range(self.stream_num):
            stream_id = i
            org, dst = self.stream[stream_id]
            interval = self.stream_packet_interval[i]
            self.packet_num[stream_id] = int(self.total_time / interval)
            packets = [Packet(j, interval * j, i, interval, org, dst) for j in range(self.packet_num[stream_id])]
            send_tasks = [SendTask(j, interval * j, stream_id, packets[j]) for j in range(self.packet_num[stream_id])]
            for j in range(self.packet_num[stream_id]):
                self.timeline.add_task(send_tasks[j], send_tasks[j].event_time)

    def cal_route_lt(self, route, lt_matrix):
        lt = 0
        for i, j in enumerate(route[0:-1]):
            org = j
            dst = route[i + 1]
            latency = lt_matrix[org, dst]
            lt = lt + latency
        lt = np.abs(lt)
        return lt

    def loadData(self):
        self.node_num = 10
        lt_folder = r'.\data'
        #  'gz-dj.csv' 'sh-dj.csv' 'simu_data4.csv' 'washedresult.csv' 'washednewresult.csv' 'noisednewresult.csv' 'finaldata.csv'
        lt_trace_name = args.lt_trace_name
        lt_path = os.path.join(lt_folder, lt_trace_name)
        lt_data = np.loadtxt(lt_path, delimiter=',', skiprows=1)
        self.tmp_matrix = lt_data.reshape((lt_data.shape[0], self.node_num, self.node_num))

    def load_lt(self, lt_path, times_tart=None, time_end=None):
        num = self.total_time / self.stream_packet_interval[0]
        if times_tart is not None and time_end is not None:
            lt_matrix = self.tmp_matrix[times_tart:time_end]
        if lt_matrix.shape[0] < num:
            print("Some Error")
            lt_matrix = np.zeros((num, self.node_num, self.node_num))
            for i in range(num):
                lt_matrix[i] = self.tmp_matrix[i % self.tmp_matrix.shape[0]]
        return lt_matrix

    def generate_route(self, send_index, packet, stream_id):
        source = packet.org
        target = packet.dst
        route_id = 0
        if self.route_alg == 'direct':
            route = [source, target]
        elif self.route_alg == 'dynamic':
            g = nx.from_numpy_array(self.lt[send_index], create_using=nx.DiGraph)
            route = nx.shortest_path(g, source, target, weight='weight')

        elif self.route_alg == 'via':
            all_sm = self.via_sms[stream_id]
            a = [sm.ucbvia(sm.mean_estimate, self.sended_packet_num[stream_id]) for sm in all_sm]
            index = np.argmin(a)
            route = self.via_toppaths[stream_id][index]
            rds = [self.cal_route_lt(route, self.via_obj_matrix[send_index + i]) for i in range(self.ptnum_per_rt)]
            rd = np.mean(rds) / self.via_ltnormals[stream_id]
            self.log.writelines(['simvalues', str(a), 'predict value', str(rds), 'rd', str(rd), '\n'])
            self.via_all_pulls[stream_id][self.sended_packet_num[stream_id] - 1] = all_sm[index].pull(rd)
            self.via_prelts[stream_id].append(np.mean(rds))
            route_id = index

        elif self.route_alg == 'vcroute':
            all_sm = self.rfvia_sms[stream_id]
            a = [sm.ucbvia(sm.mean_estimate, self.sended_packet_num[stream_id]) for sm in all_sm]
            index = np.argmin(a)
            route_id = index
            route = self.rfvia_toppaths[stream_id][index]
            self.rfvia_predict_jitter[stream_id].CalculateTargetLevel(
                packet.timestamp, packet.send_time + self.cal_route_lt(route, self.lt[send_index]))
            routeid = self.rfvia_toppaths[stream_id].index(route)

            ### 预测rd
            wan_lts = [self.cal_route_lt(route, self.lt[send_index + i]) for i in range(self.ptnum_per_rt)]
            jitters = [self.rfvia_predict_jitter[stream_id].target_level_ms_] * self.ptnum_per_rt
            y_predict = list(np.add(wan_lts, jitters))
            rd = np.mean(y_predict) / self.rfvia_ltnormals[stream_id]
            self.rfvia_prelts[stream_id].append(np.mean(wan_lts))
            self.rfvia_prejts[stream_id].append((np.mean(jitters)))
            self.log.writelines(['simvalues', str(a), 'predict value', str(y_predict), 'rd', str(rd), '\n'])
            self.rfvia_all_pulls[stream_id][self.sended_packet_num[stream_id] - 1] = all_sm[index].pull(rd)

        elif self.route_alg == 'random':
            g = nx.from_numpy_array(self.lt[send_index], create_using=nx.DiGraph)
            paths = nx.all_simple_paths(g, source, target, cutoff=3)
            paths = list(paths)
            route = paths[random.randint(0, len(paths)) - 1]
        elif self.route_alg == 'specified':
            route = self.specified_route
        return route, route_id

    def cal_ltnormal(self):
        g = nx.from_numpy_array(self.lt[0], create_using=nx.DiGraph)
        for i in range(self.stream_num):
            source, target = self.stream[i]
            simplepaths = nx.all_simple_paths(g, source, target, cutoff=3)
            simplepaths = list(simplepaths)
            paths = []
            for simplepath in simplepaths:
                if len(set(self.usrs).intersection(simplepath[1:-1])) == 0:
                    paths.append(simplepath)
            path_num = len(paths)
            preduppers = [0] * path_num
            predlowers = [0] * path_num
            for k, path in enumerate(paths):
                via_obj_hist = [0] * 100
                for j in range(100):
                    via_obj_hist[j] = self.cal_route_lt(path, self.via_obj_matrix[j])
                preduppers[k] = np.mean(via_obj_hist) + 1.96 * np.std(via_obj_hist, ddof=1) / np.sqrt(
                    np.size(via_obj_hist))
                predlowers[k] = np.mean(via_obj_hist) - 1.96 * np.std(via_obj_hist, ddof=1) / np.sqrt(
                    np.size(via_obj_hist))
            toppaths = []
            topuppers = []
            preduppermean = np.mean(preduppers)
            for k, path in enumerate(paths):
                if preduppers[k] <= preduppermean:
                    toppaths.append(path)
                    topuppers.append(preduppers[k])
            self.via_toppaths[i] = toppaths
            via_obj_normal = sum(topuppers) / len(toppaths)
            self.via_ltnormals[i] = via_obj_normal


    def rfvia_cal_ltnormal(self):
        g = nx.from_numpy_array(self.lt[0], create_using=nx.DiGraph)
        for i in range(self.stream_num):
            source, target = self.stream[i]
            simplepaths = nx.all_simple_paths(g, source, target, cutoff=3)
            simplepaths = list(simplepaths)
            paths = []
            for simplepath in simplepaths:
                if len(set(self.usrs).intersection(simplepath[1:-1])) == 0:
                    paths.append(simplepath)
            via_obj_normal = 0
            for k in range(len(paths)):
                via_obj_hist = [0] * 100
                for j in range(100):
                    via_obj_hist[j] = self.rfvia_obj_matrix[j, k]
                via_obj_upper = np.mean(via_obj_hist) + 1.96 * np.std(via_obj_hist, ddof=1) / np.sqrt(
                    np.size(via_obj_hist))
                via_obj_normal += via_obj_upper
            via_obj_normal = via_obj_normal / len(paths)
            self.rfvia_ltnormals[i] = via_obj_normal

    def cal_late_tolerate(self, stream_id):
        size = self.buffer_size[stream_id]
        if len(self.jitter[stream_id]) < size:
            newest_jitter = self.jitter[stream_id]
        else:
            newest_jitter = self.jitter[stream_id][-size:]
        abs_jitter = list(map(abs, newest_jitter))
        tolerate = np.percentile(abs_jitter, self.late_tolerant_probability[stream_id], interpolation='higher')
        return tolerate

    def alterRoute(self, begin, end):
        self.usr_loc = [begin, end]
        self.stream = [(begin, end)]

    def reset(self, time_start, time_end , t_time, r_alg, p_alg):
        self.wall_time.reset()
        self.timeline.reset()
        self.ptnum_per_rt = 1
        # 2 3 6
        self.node_num = 10
        self.usr_num = 2
        self.usrs = [0, 1, 2, 3, 4]
        self.dcs = [5, 6, 7, 8, 9]

        self.total_time = t_time

        lt_folder = r'.\data'
        #  'gz-dj.csv' 'sh-dj.csv' 'simu_data4.csv' 'washedresult.csv' 'washednewresult.csv' 'noisednewresult.csv' 'finaldata.csv'
        lt_trace_name = r'wonderdatanew.csv'

        # 'watermark' 'webRTC'
        self.process_algorithm = p_alg

        # 'direct' 'via' 'vcroute'
        self.route_alg = r_alg

        self.specified_route = [0, 1, 3, 5]

        # 'latency' 'jitter'
        self.via_optobj = 'latency'

        self.process_lt = 0
        self.stream_num = len(self.stream)
        self.stream_packet_interval = [10] * self.stream_num
        self.packet_num = [0] * self.stream_num
        self.received_packet_num = [0] * self.stream_num
        self.sended_packet_num = [0] * self.stream_num

        self.stream_wan_lt = [[] for i in range(self.stream_num)]
        self.stream_prc_lt = [[] for i in range(self.stream_num)]
        self.stream_lt = [[] for i in range(self.stream_num)]
        self.stream_loss_rate = [0] * self.stream_num
        self.stream_loss_count = [0] * self.stream_num
        self.stream_loss_count_last_period = [0] * self.stream_num
        self.stream_loss_rates = [[] for i in range(self.stream_num)]
        self.stream_output = [[] for i in range(self.stream_num)]
        self.stream_loss_id = [[] for i in range(self.stream_num)]
        # buffer state
        self.buffer = [[] for i in range(self.stream_num)]
        # the idx of max packets in the buffer
        self.buffer_size = [64] * self.stream_num
        self.buffer_free_size = [64] * self.stream_num
        self.crt_output_timestamp = [0] * self.stream_num

        # watermark state
        self.watermark = [0] * self.stream_num
        self.window_size = [1 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_size_new = [1 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_size_max = [1 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_size_min = [1 * self.stream_packet_interval[i] for i in range(self.stream_num)]
        self.window_period = [50 for i in range(self.stream_num)]
        self.window_size_ddl = [0 for i in range(self.stream_num)]
        self.loss_rate_budget = 0.01
        self.crr_packet_window = [[0, 1 * self.stream_packet_interval[i]] for i in range(self.stream_num)]
        self.latest_closed_window = [[0, 0] for i in range(self.stream_num)]
        self.latest_open_window = [[0, 0] for i in range(self.stream_num)]
        self.windows = [defaultdict(Window) for i in range(self.stream_num)]
        self.late_tolerant = [0] * self.stream_num
        self.last_delay = [0] * self.stream_num
        self.last_timestamp = [0] * self.stream_num
        self.newest_timestamp = [0] * self.stream_num
        self.jitter = [[] for i in range(self.stream_num)]
        self.late_tolerant_probability = [95] * self.stream_num
        # isolated random idx generator
        self.np_random = np.random.RandomState()

        # global timer
        self.wall_time = WallTime()

        # uses priority queue
        self.timeline = Timeline()

        self.generate_send_tasks()
        lt_path = os.path.join(lt_folder, lt_trace_name)
        self.lt = self.load_lt(lt_path, times_tart=time_start, time_end=time_end)
        self.webRTC_jitter = Jitter()

        # Via
        if self.via_optobj == 'latency':
            self.via_obj_matrix = self.lt
        if self.via_optobj == 'jitter':
            self.via_obj_matrix = np.zeros(self.lt.shape)
            self.via_obj_matrix[1:] = np.diff(self.lt, axis=0)
        self.via_ltnormals = [0] * self.stream_num
        self.via_toppaths = [None] * self.stream_num
        self.via_sms = [[] for i in range(self.stream_num)]
        self.via_all_pulls = [[0] * self.packet_num[i] for i in range(self.stream_num)]
        self.via_prelts = [[] for i in range(self.stream_num)]
        if self.route_alg == 'via':
            # 更新所有stream的via算法中的w
            self.cal_ltnormal()
            # 创建所有stream的Slot_Machine列表
            self.via_sms = [[SlotMachine() for i in range(len(self.via_toppaths[j]))] for j in range(self.stream_num)]

        # vcroute
        if self.route_alg == 'vcroute':
            self.rfvia_ltnormals = [0] * self.stream_num
            self.rfvia_toppaths = [None] * self.stream_num
            self.rfvia_sms = [[] for i in range(self.stream_num)]
            self.rfvia_all_pulls = [[0] * self.packet_num[i] for i in range(self.stream_num)]
            self.rfvia_prelts = [[] for i in range(self.stream_num)]
            self.rfvia_prejts = [[] for i in range(self.stream_num)]
            self.cal_ltnormal()
            self.rfvia_ltnormals = self.via_ltnormals
            self.rfvia_toppaths = self.via_toppaths
            self.rfvia_predict_jitter = [Jitter() for i in range(self.stream_num)]
            # 创建所有stream的Slot_Machine列表
            self.rfvia_sms = [[SlotMachine() for i in range(len(self.rfvia_toppaths[j]))] for j in
                              range(self.stream_num)]
            self.rf_output = [[0] * self.packet_num[i] for i in range(self.stream_num)]
