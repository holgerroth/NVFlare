from nvflare import SimulatorRunner

simulator = SimulatorRunner(
    job_folder="jobs/peft",
    workspace="/tmp/nvflare/nemo/peft_fedavg_345M",
    n_clients=1,
    threads=1
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
