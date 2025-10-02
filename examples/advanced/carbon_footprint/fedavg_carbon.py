import os
import pandas as pd
import pickle
import re
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg


class FedAvg(BaseFedAvg):
    def __init__(self, *args, inet_kwh_per_gb=0.01, grid_kg_per_kwh=0.475, **kwargs):
        """
        inet_kwh_per_gb: network energy intensity (kWh/GB). Default 0.01 for fixed-line.
                         For mobile you might set ~0.13; for backbone ~0.02-0.06, etc.
        grid_kg_per_kwh: grid emissions factor (kg CO2e / kWh).
        """
        super().__init__(*args, **kwargs)
        self.client_emissions = {}
        self.client_log_offsets = {}
        self.inet_kwh_per_gb = inet_kwh_per_gb
        self.grid_kg_per_kwh = grid_kg_per_kwh
        self._total_comm_bytes = 0

    def collect_emission_data(self, results):
        for result in results:
            client_name = result.meta.get("client_name")
            if not client_name:
                self.warning("Result missing client_name in meta; skipping.")
                continue

            if "EMISSIONS_DATA" not in result.meta:
                continue

            emission = result.meta["EMISSIONS_DATA"]
            emission["current_round"] = getattr(result, "current_round", None)
            self.info(f"Adding emissions data from {client_name} at round {emission['current_round']}")

            comm_bytes = emission["model_params_size"]

            # Convert B to GB
            comm_gb = comm_bytes / (1024.0 * 1024.0 * 1024.0)

            # FL formula (per round, per client): E = 2 * D_gb * Inet
            comm_energy_kwh = 2.0 * comm_gb * self.inet_kwh_per_gb
            comm_emissions_kg = comm_energy_kwh * self.grid_kg_per_kwh

            emission["comm_data_b"] = comm_bytes
            emission["comm_data_gb"] = comm_gb
            emission["comm_energy_kwh"] = comm_energy_kwh
            emission["comm_emissions_kg"] = comm_emissions_kg
            emission["inet_kwh_per_gb"] = self.inet_kwh_per_gb
            emission["grid_kg_per_kwh"] = self.grid_kg_per_kwh
            self._total_comm_bytes += comm_bytes

            self.info(
                f"[{client_name}][Round {emission['current_round']}] "
                f"+{comm_bytes:.2f} B ({comm_gb:.6f} GB), "
                f"{comm_energy_kwh:.6f} kWh, {comm_emissions_kg:.6f} kgCO2e "
                f"(Inet={self.inet_kwh_per_gb} kWh/GB, factor x2)"
            )

            if client_name not in self.client_emissions:
                self.client_emissions[client_name] = [emission]
            else:
                self.client_emissions[client_name].append(emission)

        self.info(f"Added emissions data to client_emissions {len(self.client_emissions)}")

    def parse_comm_since_last(self, client_name, log_path, offset):
        pattern = re.compile(r"size:.*\((\d+) Bytes\)")
        total_bytes = 0

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(offset)
                for line in f:
                    match = pattern.search(line)
                    if match:
                        total_bytes += int(match.group(1))
                new_offset = f.tell()
            return total_bytes / 1024.0, new_offset  # KB and updated offset
        except FileNotFoundError:
            self.warning(f"[CommLogs] Missing log file for {client_name}: {log_path}")
            return 0.0, offset
        except Exception as e:
            self.warning(f"[CommLogs] Failed to parse {client_name} log {log_path}: {e}")
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
            self.collect_emission_data(results)

            aggregate_results = self.aggregate(results)
            model = self.update_model(model, aggregate_results)
            self.save_model(model)

        self.info("Finished FedAvg.")

        self.info(f"Received emissions from {len(self.client_emissions)} clients.")
        for client_name, emissions in self.client_emissions.items():
            self.info(f"Client {client_name}: {len(emissions)} records.")

        # Final FL total using your formula over the whole run:
        total_gb = (self._total_comm_bytes / (1024.0 ** 3))
        total_energy_kwh = 2.0 * total_gb * self.inet_kwh_per_gb
        total_emissions_kg = total_energy_kwh * self.grid_kg_per_kwh
        self.info(
            f"[FL Total] data={total_gb:.6f} GB, energy={total_energy_kwh:.6f} kWh, "
            f"emissions={total_emissions_kg:.6f} kgCO2e "
            f"(Inet={self.inet_kwh_per_gb} kWh/GB, factor x2)"
        )

        self.save_client_emissions()

    def save_client_emissions(self):
        with open("client_emissions.pkl", "wb") as f:
            pickle.dump(self.client_emissions, f)
        self.info(f"Saved all client emissions to {os.path.join(os.getcwd(), 'client_emissions.pkl')}")

        out_client_emissions = {
            "round": [],
            "timestamp": [],
            "client": [],
            # Train (CodeCarbon "train" task)
            "emissions": [],
            "cpu_energy": [],
            "gpu_energy": [],
            "ram_energy": [],
            "energy_consumed": [],
            # Communication (GB-based only)
            "comm_data_gb": [],
            "comm_energy_kwh": [],
            "comm_emissions_kg": [],
            "inet_kwh_per_gb": [],
            "grid_kg_per_kwh": [],
            # Idle (CodeCarbon "idle" task)
            "idle_timestamp": [],
            "idle_emissions": [],
            "idle_cpu_energy": [],
            "idle_gpu_energy": [],
            "idle_ram_energy": [],
            "idle_energy_consumed": [],
            "idle_duration_sec": [],
        }

        def _get(x, name, alt=None):
            try:
                return getattr(x, name)
            except Exception:
                try:
                    return x.get(name, alt)
                except Exception:
                    return alt

        for client_name, emissions in self.client_emissions.items():
            for emission in emissions:
                train = emission.get("train")
                idle = emission.get("idle")
                out_client_emissions["round"].append(emission.get("current_round"))
                out_client_emissions["client"].append(client_name)

                # Train metrics
                if train is None:
                    out_client_emissions["timestamp"].append(None)
                    out_client_emissions["emissions"].append(None)
                    out_client_emissions["cpu_energy"].append(None)
                    out_client_emissions["gpu_energy"].append(None)
                    out_client_emissions["ram_energy"].append(None)
                    out_client_emissions["energy_consumed"].append(None)
                else:
                    out_client_emissions["timestamp"].append(_get(train, "timestamp"))
                    out_client_emissions["emissions"].append(_get(train, "emissions"))
                    out_client_emissions["cpu_energy"].append(_get(train, "cpu_energy"))
                    out_client_emissions["gpu_energy"].append(_get(train, "gpu_energy"))
                    out_client_emissions["ram_energy"].append(_get(train, "ram_energy"))
                    out_client_emissions["energy_consumed"].append(_get(train, "energy_consumed"))

                # Communication (GB only)
                out_client_emissions["comm_data_gb"].append(emission.get("comm_data_gb", 0.0))
                out_client_emissions["comm_energy_kwh"].append(emission.get("comm_energy_kwh", 0.0))
                out_client_emissions["comm_emissions_kg"].append(emission.get("comm_emissions_kg", 0.0))
                out_client_emissions["inet_kwh_per_gb"].append(emission.get("inet_kwh_per_gb", self.inet_kwh_per_gb))
                out_client_emissions["grid_kg_per_kwh"].append(emission.get("grid_kg_per_kwh", self.grid_kg_per_kwh))

                # Idle metrics
                if idle is None:
                    out_client_emissions["idle_timestamp"].append(None)
                    out_client_emissions["idle_emissions"].append(None)
                    out_client_emissions["idle_cpu_energy"].append(None)
                    out_client_emissions["idle_gpu_energy"].append(None)
                    out_client_emissions["idle_ram_energy"].append(None)
                    out_client_emissions["idle_energy_consumed"].append(None)
                    out_client_emissions["idle_duration_sec"].append(None)
                else:
                    out_client_emissions["idle_timestamp"].append(_get(idle, "timestamp"))
                    out_client_emissions["idle_emissions"].append(_get(idle, "emissions"))
                    out_client_emissions["idle_cpu_energy"].append(_get(idle, "cpu_energy"))
                    out_client_emissions["idle_gpu_energy"].append(_get(idle, "gpu_energy"))
                    out_client_emissions["idle_ram_energy"].append(_get(idle, "ram_energy"))
                    out_client_emissions["idle_energy_consumed"].append(_get(idle, "energy_consumed"))
                    duration = _get(idle, "duration")
                    if duration is None:
                        duration = _get(idle, "duration_sec", _get(idle, "duration_seconds"))
                    out_client_emissions["idle_duration_sec"].append(duration)

        # Save to CSV
        pd.DataFrame(out_client_emissions).to_csv("client_emissions.csv", index=False)
        self.info(f"Saved select client emissions to {os.path.join(os.getcwd(), 'client_emissions.csv')}")