import heapq
import itertools


# <事件，时间>，<事件，时间>，<事件，时间>，<事件，时间>，<事件，时间>

class Timeline(object):
    def __init__(self):
        # priority queue
        self.pq = []
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'
        # tie breaker
        self.counter = itertools.count()

    def add_task(self, task, priority=0):
        # 'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        # 'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        # 'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def __len__(self):
        return len(self.pq)

    def reset(self):
        self.pq = []
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()
