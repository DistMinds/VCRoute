class Packet(object):
    def __init__(self, idx, timestamp, stream_id, size, origin, destination):
        self.idx = idx
        self.timestamp = timestamp
        self.stream_id = stream_id
        self.size = size
        self.org = origin
        self.dst = destination
        self.send_time = None
        self.arrival_time = None
        self.output_time = None
        self.route = None
        self.route_id = None
        self.window_start = None
        self.window_end = None
        # self.send_info[timestamp,wan_lt,window_end,window_end-timestamp,watermark,jitter,ete_lt]
        self.send_info = [timestamp, 0, 0, 0, 0, 0, 0]

    def set_send_time(self, send_time):
        self.send_time = send_time

    def set_arrival_time(self, arrival_time):
        self.arrival_time = arrival_time

    def set_output_time(self, output_time):
        self.output_time = output_time

    def set_route(self, route):
        self.route = route

    def set_window(self, start, end):
        self.window_start = start
        self.window_end = end


class Window(object):
    def __init__(self, stream_id, size, start_time, end_time, packets):
        # self.idx = idx
        self.stream_id = stream_id
        self.size = size
        self.start_time = start_time
        self.end_time = end_time
        self.packets = packets

    def set_window_time(self, start, end):
        self.start_time = start
        self.end_time = end

    def append_packet(self, packet):
        self.packets.append(packet)

    def sort_packets(self):
        self.packets = sorted(self.packets, key=lambda packet: packet.timestamp)
