# Welcome to VCRoute
The need for having global audio/video meetings is ever increasing, which causes more traffic to be sent across geo-distributed DCs. Inter-DC network performance is limited and highly dynamic, which causes latency problem to existing video conferencing systems. To provide low-latency video conferencing services, we identified two deficiencies of existing systems. First, existing routing methods in video conferencing systems do not consider the packet reordering latency when selecting routes, thus may lead to high end-to-end latency. We proposed a video conferencing application specific router named \emph{VCRoute}, which jointly considers the data transmitting and reordering time to find the best paths for video streams. Second, existing buffer-based jitter management methods inevitably cause extra delay for packets in the buffer. To avoid this issue, we leverage the watermark-based out-of-order processing (OOP) mechanism to further reduce the latency. Evaluations on real geo-distributed environments demonstrate the effectiveness and efficiency of the proposed techniques.


# Getting Started

## Dependency:

- Python 3.8 or higher

## Deploying VCRoute
Firstly,You can place the dataset in the `data` folder. We also place two dataset in the `data` folder:
| Dataset | Description |
| ------ | ------ |
| finaldata.csv | Network latency of 600,000 packets between 10 cities |
| wonderdatanew.csv | Network latency of 30,000 packets between 10 cities |

Make sure that all the dependencies are installed and that the versions of dependencies rely on meet the requirements, then you need to run the following command:
```bash
git clone https://github.com/DistMinds/VCRoute.git
```

# Running an example

There are a number of parameters that need to be specified. You can get the details of the parameters using the following command
```bash
python main.py -h
```
the result is as follow:
```bash
usage: main.py [-h] [--lt_trace_name LT_TRACE_NAME] [--total_time TOTAL_TIME] [--route_alg ROUTE_ALG]
               [--process_alg PROCESS_ALG] [--start_node START_NODE] [--end_node END_NODE] [--begin_index BEGIN_INDEX]
               [--end_index END_INDEX]
optional arguments:
  -h, --help            show this help message and exit
  --lt_trace_name LT_TRACE_NAME
                        name of the dataset file
  --total_time TOTAL_TIME
                        total time of streaming
  --route_alg ROUTE_ALG
                        route algorithm: direct, via or vcroute
  --process_alg PROCESS_ALG
                        process algorithm: webrtc or watermark
  --start_node START_NODE
                        the start node index
  --end_node END_NODE   the end node index
  --begin_index BEGIN_INDEX
                        the index of first packet
  --end_index END_INDEX
                        the index of last packet + 1
```
We also set a default value of these parameters:
| Parameter | Value |
| ------ | ------ |
| lt_trace_name | wonderdatanew.csv |
| total_time | 300000 |
| route_alg | vcroute |
| process_alg | watermark |
| start_node | 0 |
| end_node | 5 |
| begin_index | 0 |
| end_index | 30000 |

So you can run an example just use the following command:
```bash
python main.py
```
the results are all saved in the `result` folder. It mainly has two result files.
 - eleltxx.csv
    | end to end latancy | process latancy |
    | ------ | ------ |
 - routexx.txt
    | packet index | route | end to end latancy | process latancy |
    | ------ | ------ | ------ | ------ |
