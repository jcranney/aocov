import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme()

with open("results.txt","r") as f:
    lines = f.readlines()

lines = [line.split("|")[1:-1] for line in lines][2:]

aotools = [float(line[-1]) for line in lines if ("cpu" in line[1]) and ("no pytorch" in line[2])]
aocovcpu = [float(line[-1]) for line in lines if ("cpu" in line[1]) and ("all in aocov" in line[2])]
aocovgpu = [float(line[-1]) for line in lines if ("cuda" in line[1]) and ("all in aocov" in line[2])]

n = np.array([int(line[0]) for line in lines if "no pytorch" in line[2]])
w = 2
plt.bar(n-w,aotools,w,label="aotools")
plt.bar(n,aocovcpu,w,label="aocov cpu")
plt.bar(n+w,aocovgpu,w,label="aocov gpu")
plt.xlabel("n (width of grid)")
plt.ylabel("time to solution [sec]")
plt.yscale("log")
plt.tight_layout()
plt.savefig("performance.png",dpi=300,transparent=False)