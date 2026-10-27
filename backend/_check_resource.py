import json
import math
import sys

from services import resource_monitor as rm

snapshot = rm.resource_snapshot()
print(json.dumps(snapshot, indent=1))
with rm.ResourceTracker("demo") as tracker:
    sum(i * i for i in range(200_000))
print(json.dumps(tracker.usage.as_dict(), indent=1))
print("violations:", rm.check_hard_limits(1000, 10, 0.4))
print("big:", rm.check_hard_limits(10_000_000, 10, 0.4))
