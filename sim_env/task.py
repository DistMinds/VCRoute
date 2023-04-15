
class SendTask(object):
    def __init__(self, idx, event_time, stream_id, packet):
        self.idx = idx
        self.event_time = event_time
        self.stream_id = stream_id
        self.packet = packet


class ReceiveTask(object):
    def __init__(self, idx, event_time, stream_id, packet):
        self.idx = idx
        self.event_time = event_time
        self.stream_id = stream_id
        self.packet = packet
