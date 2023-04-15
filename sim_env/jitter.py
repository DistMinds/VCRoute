from collections import deque
import sys


class Packet(object):
    def __init__(self, n, b, e, r, route):
        self.idx = n
        self.bagsize = b
        self.eventtime = e
        self.receive_time = r
        self.prc_lt = 0
        self.totaldelay = 0
        self.route = route


class PacketArrivedInfo:
    def __init__(self, mt, ms, bf, ma):
        self.main_timestamp = mt
        self.main_sequeuce_number = ms
        self.buffer_flush = bf
        self.main_arrivaltime = ma


class PacketDelay:
    def __init__(self, iat, t):
        self.iat_delay_ms = iat
        self.timestamp = t


class Histogram:
    def __init__(self, num_buckets, forget_factor, start_forget_weight):
        self.buckets_ = []
        i = 0
        while i < num_buckets:
            self.buckets_.append(0)
            i = i + 1
        self.forget_factor_ = 0
        self.base_forget_factor_ = forget_factor
        self.add_count_ = 0
        self.start_forget_weight_ = start_forget_weight

    def Add(self, value):
        vector_sum = 0
        bucket_i = 0
        while bucket_i < len(self.buckets_):
            self.buckets_[bucket_i] = int(self.buckets_[bucket_i] * self.forget_factor_) >> 15
            vector_sum += self.buckets_[bucket_i]
            bucket_i = bucket_i + 1
        '''for bucket in self.buckets_:
            bucket= int(bucket * self.forget_factor_) >> 15
            vector_sum += bucket'''
        self.buckets_[int(value)] += int(32768 - self.forget_factor_) << 15
        vector_sum += int(32768 - self.forget_factor_) << 15
        vector_sum -= 1 << 30
        if vector_sum != 0:
            flip_sign = -1 if vector_sum > 0 else 1
            for bucket in self.buckets_:
                correction = flip_sign * min(abs(vector_sum), bucket >> 4)
                bucket += correction
                vector_sum += correction
                if abs(vector_sum) == 0:
                    break
        self.add_count_ = self.add_count_ + 1

        if self.start_forget_weight_ != 0:
            if self.forget_factor_ != self.base_forget_factor_:
                old_forget_factor = self.forget_factor_
                forget_factor = (1 << 15) * (1 - self.start_forget_weight_ / (self.add_count_ + 1))
                self.forget_factor_ = max(0, min(self.base_forget_factor_, forget_factor))
        else:
            self.forget_factor_ += (self.base_forget_factor_ - self.forget_factor_ + 3) >> 2

    def Quantile(self, probability):
        inverse_probability = (1 << 30) - probability
        index = 0
        sum = 1 << 30
        sum -= self.buckets_[index]

        while sum > inverse_probability and index < len(self.buckets_) - 1:
            index = index + 1
            sum -= self.buckets_[index]
        return index


class Jitter(object):
    def __init__(self):
        self.jitter_buffer_packets_received = 0
        self.first_packet_ = True
        self.delay_history_ = deque()
        self.packet_buffer = deque(maxlen=200)
        i = 0
        while i < 200:
            self.packet_buffer.append(Packet(0, 0, 0, 0, None))
            i += 1
        self.outputData = deque()
        self.number_of_packet = 0
        self.timestamp_ = 0
        self.kStartDelayMs = 80
        self.kDelayBuckets = 100
        #1 or 2
        self.target_level_multiplier = 1
        self.target_level_threshold_ms = 450
        self.sample_rate = 48000
        self.max_number_of_packets_ = 200
        self.max_packets_in_buffer_ = 200
        self.effective_minimum_delay_ms_ = 0
        self.last_timestamp_ = 0
        self.last_arrivaltime_ = 0
        self.max_history_ms = 2000
        self.newest_timestamp_ = 0
        self.kBucketSizeMs = 10  # 要和packet size一致
        self.maximum_delay_ms_ = 4000
        self.packet_len_ms_ = 10  # 要和packet size一致
        self.forget_factor = 0.983
        self.start_forget_weight = 2
        self.target_level_ms_ = self.kStartDelayMs
        # self.target_level_ms_ = 0
        self.target_level = self.target_level_ms_ / self.kBucketSizeMs
        self.insert_index = 0
        self.histogram_ = Histogram(self.kDelayBuckets, (self.forget_factor * (1 << 15)), self.start_forget_weight)
        self.optimal_delay_ms_1 = 0
        self.optimal_delay_ms_2 = 0
        self.ms_per_loss_percent_ = 20

    def UpdateDelayHistory(self, iat_delay_ms, timestamp):
        delay = PacketDelay(iat_delay_ms, timestamp)
        self.delay_history_.append(delay)
        while (timestamp - self.delay_history_[0].timestamp) > self.max_history_ms:
            self.delay_history_.popleft()

    def CalculateRelativePacketArrivalDelay(self):
        relative_delay = 0
        for delay in self.delay_history_:
            relative_delay += delay.iat_delay_ms
            relative_delay = max(relative_delay, 0)
        return relative_delay

    def CalculateRelativeDelay(self, timestamp, arrival_time):
        if self.last_arrivaltime_ == 0:
            self.last_arrivaltime_ = arrival_time
            self.newest_timestamp_ = timestamp
            self.last_timestamp_ = timestamp
        expected_iat_ms = timestamp - self.last_timestamp_
        iat_ms = arrival_time - self.last_arrivaltime_
        iat_delay_ms = iat_ms - expected_iat_ms
        self.UpdateDelayHistory(iat_delay_ms, timestamp)
        relative_delay = self.CalculateRelativePacketArrivalDelay()
        self.last_timestamp_ = timestamp
        self.last_arrivaltime_ = arrival_time
        if timestamp > self.newest_timestamp_:
            self.newest_timestamp_ = timestamp
        return relative_delay

    def CalculateTargetLevel1(self, relative_delay_ms):
        histogram_update = relative_delay_ms
        index = histogram_update / self.kBucketSizeMs
        if index < self.kDelayBuckets:
            self.histogram_.Add(index)
        bucket_index = self.histogram_.Quantile(0.95 * (1 << 30))
        self.optimal_delay_ms_1 = (1 + bucket_index) * self.kBucketSizeMs

    def MinimizeCostFunction(self, base_delay_ms):
        buckets = self.histogram_.buckets_
        loss_probability = 1 << 30
        min_cost = sys.maxsize
        min_bucket = 0
        for i in range(0, len(buckets)):
            loss_probability -= buckets[i]
            delay_ms = max(0, i * self.kBucketSizeMs - base_delay_ms) << 30
            cost = delay_ms + 100 * self.ms_per_loss_percent_ * loss_probability
            if cost < min_cost:
                min_cost = cost
                min_bucket = i
            if loss_probability == 0:
                break
        return min_bucket

    def CalculateTargetLevel2(self, relative_delay_ms, reordered, base_delay_ms):
        index = relative_delay_ms / self.kBucketSizeMs if reordered else 0
        if index < self.kDelayBuckets:
            self.histogram_.Add(index)
        bucket_index = self.MinimizeCostFunction(base_delay_ms)
        self.optimal_delay_ms_2 = (1 + bucket_index) * self.kBucketSizeMs

    def CalculateTargetLevel(self, timestamp, arrivaltime):
        relative_delay = self.CalculateRelativeDelay(timestamp, arrivaltime)
        reordered = self.newest_timestamp_ != timestamp
        if not reordered:
            self.CalculateTargetLevel1(relative_delay)
        self.target_level_ms_ = self.optimal_delay_ms_1
        if reordered:
            self.CalculateTargetLevel2(relative_delay, reordered, self.target_level_ms_)
            self.target_level_ms_ = max(self.target_level_ms_, self.optimal_delay_ms_2)
        self.target_level_ms_ = max(self.target_level_ms_, self.effective_minimum_delay_ms_)
        if self.maximum_delay_ms_ > 0:
            self.target_level_ms_ = min(self.target_level_ms_, self.maximum_delay_ms_)
        if self.packet_len_ms_ > 0:
            self.target_level_ms_ = max(self.target_level_ms_, self.packet_len_ms_)
            self.target_level_ms_ = min(self.target_level_ms_,
                                        3 * self.max_packets_in_buffer_ * self.packet_len_ms_ / 4)
        self.target_level = self.target_level_ms_ / self.kBucketSizeMs
        return relative_delay

    def PartialFlush(self, target_level_ms, main_arrivaltime):
        '''target_level_samples = min(target_level_ms, max_number_of_packets_ * kBucketSizeMs / 2)'''
        '''target_level_samples = max(target_level_samples, target_level_threshold_ms)'''
        target_level_samples = target_level_ms
        max_receive_time = 0
        for i in range(0, self.number_of_packet):
            if self.packet_buffer[i].receive_time > max_receive_time:
                max_receive_time = self.packet_buffer[i].receive_time
        while self.number_of_packet * self.kBucketSizeMs > target_level_samples or self.number_of_packet > self.max_number_of_packets_ / 2:
            temp = self.packet_buffer.popleft()
            self.packet_buffer.append(temp)
            temp.prc_lt += main_arrivaltime - temp.receive_time
            temp.totaldelay += main_arrivaltime - temp.eventtime
            self.outputData.append(temp)
            self.number_of_packet = self.number_of_packet - 1

    def Flush(self, main_arrivaltime):
        max_receive_time = 0
        for i in range(0, self.number_of_packet):
            if self.packet_buffer[i].receive_time > max_receive_time:
                max_receive_time = self.packet_buffer[i].receive_time
        while self.number_of_packet > 0:
            temp = self.packet_buffer.popleft()
            self.packet_buffer.append(temp)
            temp.prc_lt += main_arrivaltime - temp.receive_time
            temp.totaldelay += main_arrivaltime - temp.eventtime
            self.outputData.append(temp)
            self.number_of_packet = self.number_of_packet - 1

    def InsertPacket(self, packet, target_level_ms, main_arrivaltime):
        return_val = 222
        span_threshold = self.target_level_multiplier * target_level_ms
        '''max(target_level_threshold_ms, target_level_ms);'''
        smart_flush = (self.number_of_packet * packet.bagsize) >= span_threshold
        if self.number_of_packet >= self.max_number_of_packets_ or smart_flush:
            buffer_size_before_flush = self.number_of_packet
            if smart_flush:
                self.PartialFlush(target_level_ms, main_arrivaltime)
                return_val = 111  # kPartialFlush
            else:
                self.Flush(main_arrivaltime)
                return_val = 333  # kFlushed
            print("Packet buffer flushed, " + "target buffer size is " + str(self.target_level)+', '+str(
                buffer_size_before_flush - self.number_of_packet) + " packets output. now the idx of packet is " + str(
                self.number_of_packet) + "\n")

        rit = 0
        while rit < self.number_of_packet:
            if self.packet_buffer[rit].eventtime > packet.eventtime:
                break
            rit = rit + 1
        if rit != self.number_of_packet:
            for i in range(self.number_of_packet - 1, rit - 1, -1):
                self.packet_buffer[i + 1] = self.packet_buffer[i]
        self.insert_index = rit
        self.packet_buffer[rit] = packet
        self.number_of_packet = self.number_of_packet + 1
        return return_val

    def InsertPacketInternal(self, idx, size, timestamp, arrival_time, route):
        packet = Packet(idx, size, timestamp, arrival_time, route)
        self.jitter_buffer_packets_received = self.jitter_buffer_packets_received + 1
        update_sample_rate_and_channels = self.first_packet_
        main_timestamp = packet.eventtime
        main_sequence_number = packet.idx
        main_arrivaltime = packet.receive_time
        ret = self.InsertPacket(packet, self.target_level_ms_, main_arrivaltime)
        buffer_flush_occurred = False
        if ret == 333:
            update_sample_rate_and_channels = True
            buffer_flush_occurred = True
        elif ret == 111:
            self.timestamp_ = (0 if self.number_of_packet == 0 else self.packet_buffer[0].receive_time)
            buffer_flush_occurred = True
        elif ret != 222:
            return 555
        info = PacketArrivedInfo(main_timestamp, main_sequence_number, buffer_flush_occurred, main_arrivaltime)
        relative_delay = self.CalculateTargetLevel(info.main_timestamp, info.main_arrivaltime)
        return 0
