try:
    import waitGPU
except ImportError:
    print('Failed to import waitGPU --> no automatic scheduling on GPU')
    waitGPU = None
    pass
import subprocess
import time
import os


def run_command(command, on_gpu, noprint):
    if not on_gpu:
        while True:
            time.sleep(1)
            if os.getloadavg()[0] < 4:
                break
    elif waitGPU is not None:
        try:
            waitGPU.wait(nproc=0, interval=1, ngpu=1, gpu_ids=[0, 1, 2, 3])
        except:
            print("Failed to run `waitGPU.wait` --> no automatic scheduling on GPU")
    command = " ".join(command.split())
    if noprint:
        command = "{} > /dev/null".format(command)
    print(command)
    subprocess.Popen(command, stderr=subprocess.STDOUT, stdout=None, shell=True)


def launch(jobs, interval=5, on_gpu=True, no_print=True):
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        run_command(job, on_gpu, no_print)
        time.sleep(interval)
