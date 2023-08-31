from agsi.ReportAnalysis import FullAnalysis
import numpy as np
from time import perf_counter

t1 = perf_counter()

output = FullAnalysis('input.npy', specific_bound=True)

t2 = perf_counter()
print(f"\n # RUNTIME: {(t2 - t1):.1f} s")

np.save('output_name.npy', output)
