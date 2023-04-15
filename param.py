import argparse

parser = argparse.ArgumentParser(description='streaming')

# -- Basic --
parser.add_argument('--lt_trace_name', type=str, default='wonderdatanew.csv',
                    help='name of the dataset file')

# -- Environment --
parser.add_argument('--total_time', type=int, default=300000,
                    help='total time of streaming')
parser.add_argument('--route_alg', type=str, default='vcroute',
                    help='route algorithm: direct, via or vcroute')
parser.add_argument('--process_alg', type=str, default='watermark',
                    help='process algorithm: webrtc or watermark')
parser.add_argument('--start_node', type=int, default=0,
                    help='the start node index')
parser.add_argument('--end_node', type=int, default=5,
                    help='the end node index')
parser.add_argument('--begin_index', type=int, default=0,
                    help='the index of first packet')
parser.add_argument('--end_index', type=int, default=30000,
                    help='the index of last packet + 1')

args = parser.parse_args()
