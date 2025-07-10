import os
import pandas as pd
import pickle
import re
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg

class FedAvg(BaseFedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_emissions = {}
        self.client_log_offsets = {}

    def collect_emission_data(self, results):
        """Collect emission data from the meta information of results.
        
        Args:
            results: List of results from clients containing meta information
            
        Returns:
            dict: Aggregated emission data across all clients
        """

        for result in results:
            client_name = result.meta['client_name']
            if 'EMISSIONS_DATA' in result.meta:
                emission = result.meta['EMISSIONS_DATA']
                emission["current_round"] = result.current_round
                self.info(f"Adding emissions data from {client_name} at round {result.current_round}")

                # NEW: Compute communication energy for this round only
                log_path = f"/groups/lingurarugrp/atapp/NVFlare/examples/advanced/carbon_footprint/run/{client_name}/log.txt"
                offset = self.client_log_offsets.get(client_name, 0)

                comm_kb, new_offset = self.parse_comm_since_last(client_name, log_path, offset)
                comm_energy = comm_kb * 6.894e-8  # kWh
                comm_emissions = comm_energy * 0.475  # kgCO2

                emission["comm_data_kb"] = comm_kb
                emission["comm_energy"] = comm_energy
                emission["comm_emissions"] = comm_emissions

                self.client_log_offsets[client_name] = new_offset
                self.info(
                    f"[{client_name}][Round {result.current_round}] +{comm_kb:.2f}kB, {comm_energy:.6f}kWh, {comm_emissions:.6f}kgCO2")

                # Save emissions
                if client_name not in self.client_emissions:
                    self.client_emissions[client_name] = [emission]
                else:
                    self.client_emissions[client_name].append(emission)

        self.info(f"Added emissions data to client_emissions {len(self.client_emissions)}")

    def parse_comm_since_last(self, client_name, log_path, offset):
        """Parse new communication entries from log file since last offset."""
        pattern = re.compile(r"size:.*\((\d+) Bytes\)")
        total_bytes = 0

        try:
            with open(log_path, "r") as f:
                f.seek(offset)
                for line in f:
                    match = pattern.search(line)
                    if match:
                        total_bytes += int(match.group(1))
                new_offset = f.tell()
            return total_bytes / 1024.0, new_offset  # Return KB and updated offset
        except FileNotFoundError:
            self.warning(f"[CommLogs] Missing log file for {client_name}")
            return 0.0, offset

    def run(self) -> None:
        self.info("Start FedAvg.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            results = self.send_model_and_wait(targets=clients, data=model)
            
            # Collect emission data from results
            self.collect_emission_data(results)

            aggregate_results = self.aggregate(results)

            model = self.update_model(model, aggregate_results)

            self.save_model(model)

        self.info("Finished FedAvg.")

        self.info(f"Received emissions from {len(self.client_emissions)} clients.")
        for client_name, emissions in self.client_emissions.items():
            self.info(f"Client {client_name}: {len(emissions)} records.")
        self.save_client_emissions()

    def save_client_emissions(self):

        with open('client_emissions.pkl', 'wb') as f:
            pickle.dump(self.client_emissions, f)
        self.info(f"Saved all client emissions to {os.path.join(os.getcwd(), 'client_emissions.pkl')}")

        # Prepare selected fields for CSV export
        out_client_emissions = {
            "round": [],
            "timestamp": [],
            "client": [],
            "emissions": [],
            "cpu_energy": [],
            "gpu_energy": [],
            "ram_energy": [],
            "energy_consumed": [],
            "comm_data_kb": [],
            "comm_energy": [],
            "comm_emissions": [],
        }

        for client_name, emissions in self.client_emissions.items():
            for emission in emissions:
                e = emission["train"]
                out_client_emissions["round"].append(emission["current_round"])
                out_client_emissions["client"].append(client_name)
                out_client_emissions["timestamp"].append(e.timestamp)

                out_client_emissions["emissions"].append(e.emissions)
                out_client_emissions["cpu_energy"].append(e.cpu_energy)
                out_client_emissions["gpu_energy"].append(e.gpu_energy)
                out_client_emissions["ram_energy"].append(e.ram_energy)
                out_client_emissions["energy_consumed"].append(e.energy_consumed)

                # Add communication emissions if present; else default to 0
                out_client_emissions["comm_data_kb"].append(emission.get("comm_data_kb", 0))
                out_client_emissions["comm_energy"].append(emission.get("comm_energy", 0))
                out_client_emissions["comm_emissions"].append(emission.get("comm_emissions", 0))

        # Save to CSV
        pd.DataFrame(out_client_emissions).to_csv("client_emissions.csv", index=False)
        self.info(f"Saved select client emissions to {os.path.join(os.getcwd(), 'client_emissions.csv')}")
