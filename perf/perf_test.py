import numpy as np
import aotools
import aocov
import timeit

if __name__ == "__main__":
    r0 = 0.15
    L0 = 25.0

    def print_perf(n, dev, exp, r):
        print(f"| {n:4d} | {dev:8s} | {exp:30s} | {r:10.3e} |")

    print(f"| {'n':4s} | {'device':8s} | {'experiment':30s} | {'sec/matrix':10s} |")
    print(f"|-{'-'*4}-|-{'-'*8}-|-{'-'*30}-|-{'-'*10}-|")
    for n in [16,32,64,128]:
        
        xx, yy = np.meshgrid(
            np.linspace(-4,4,n),
            np.linspace(-4,4,n),
            indexing="xy"
        )
        xx = xx.flatten()
        yy = yy.flatten()

        device = "cpu"
        def func():
            rr = ((xx[:, None]-xx[None, :])**2 + (yy[:, None]-yy[None, :])**2)**0.5
            cov = aotools.phase_covariance(rr,r0,L0)
            return cov
        r = timeit.timeit(func, number=10)
        exp_str = "no pytorch"
        print_perf(n,device,exp_str,r)

        device = "cpu"
        def func():
            rr = ((xx[:, None]-xx[None, :])**2 + (yy[:, None]-yy[None, :])**2)**0.5
            cov = aocov.phase_covariance(rr,r0,L0,device=device)
            return cov    
        r = timeit.timeit(func, number=10)
        exp_str = "rr in numpy, rest in aocov"
        print_perf(n,device,exp_str,r)

        device = "cpu"
        def func():
            cov = aocov.phase_covariance_xyxy(xx,yy,xx,yy,r0,L0,device=device), 
            return cov    
        r = timeit.timeit(func, number=10)
        exp_str = "all in aocov"
        print_perf(n,device,exp_str,r)
        
        try:
            device = "cuda:0"
            def func():
                rr = ((xx[:, None]-xx[None, :])**2 + (yy[:, None]-yy[None, :])**2)**0.5
                cov = aocov.phase_covariance(rr,r0,L0,device=device)
                return cov
            r = timeit.timeit(func, number=10)
            exp_str = "rr in numpy, rest in aocov"
            print_perf(n,device,exp_str,r)
        except RuntimeError as e:
            pass
            # print("No cuda:0 device found, skipping.")

        try:
            device = "cuda:0"
            def func():
                cov = aocov.phase_covariance_xyxy(xx,yy,xx,yy,r0,L0,device=device)
                return cov
            r = timeit.timeit(func, number=10)
            exp_str = "all in aocov"
            print_perf(n,device,exp_str,r)
        except RuntimeError as e:
            pass
            #print("No cuda:0 device found, skipping.")
        