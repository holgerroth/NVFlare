# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from flwr.server.strategy import Strategy as FlwrStrategy
from flwr.server.client_proxy import ClientProxy
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg


class FlwrController(BaseFedAvg):
    def __init__(self, strategy: FlwrStrategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
    
    def update_model_with_parameters(self, model, parameters):
        """Update model with aggregated parameters from strategy"""
        model.set_weights(parameters)
        return model
    
    def run(self) -> None:
        self.info("Start FedAvg with Flower Strategy.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            # Get current model parameters
            current_parameters = model.get_weights()
            
            # Configure fit instructions using the strategy
            fit_ins = self.strategy.configure_fit(
                server_round=self.current_round,
                parameters=current_parameters,
                client_manager=None  # Will be handled by sample_clients
            )
            
            # Sample clients for this round
            clients = self.sample_clients(self.num_clients)
            self.info(f"Sampled {len(clients)} clients for round {self.current_round}")

            # Send fit instructions to clients and collect results
            fit_results = []
            for client in clients:
                # Send model with fit instructions
                result = self.send_model_and_wait(targets=[client], data={
                    'model': model,
                    'fit_ins': fit_ins
                })
                
                if result and len(result) > 0:
                    # Convert result to expected format for strategy
                    client_result = result[0]  # Get result from first (and only) client
                    fit_results.append((
                        ClientProxy(cid=client.name),
                        client_result
                    ))

            self.info(f"Collected {len(fit_results)} fit results")

            # Aggregate results using the strategy
            if fit_results:
                aggregate_result = self.strategy.aggregate_fit(
                    server_round=self.current_round,
                    results=fit_results,
                    failures=[]
                )
                
                if aggregate_result is not None:
                    # Update model with aggregated parameters
                    aggregated_parameters, metrics = aggregate_result
                    model = self.update_model_with_parameters(model, aggregated_parameters)
                    
                    # Log aggregation metrics if available
                    if metrics:
                        self.info(f"Aggregation metrics: {metrics}")
                else:
                    self.warning("Strategy returned None for aggregate_fit, skipping model update")
            else:
                self.warning("No fit results to aggregate, skipping round")

            # Save model after each round
            self.save_model(model)

        self.info("Finished FedAvg with Flower Strategy.")
