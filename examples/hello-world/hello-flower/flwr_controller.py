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


from typing import Dict, List, Optional, Tuple
import time

from flwr.common import Parameters, Status, Code, FitIns, FitRes, EvaluateIns, EvaluateRes, GetParametersIns, GetParametersRes, GetPropertiesIns, GetPropertiesRes, ReconnectIns, ndarray_to_bytes, bytes_to_ndarray
from flwr.server.client_manager import ClientManager as FlwrClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy as FlwrStrategy

from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_common.abstract.fl_model import FLModel


class NVFlareClientProxy(ClientProxy):
    """Adapter class for NVFlare clients to work with Flower's ClientProxy interface."""
    
    def __init__(self, cid: str, result=None):
        super().__init__(cid=cid)
        self.result = None
    
    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Get properties of the client."""
        # Return empty properties as this is just an adapter
        return GetPropertiesRes(properties={})
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get parameters from the client."""
        # This might need proper implementation if used
        return GetParametersRes(parameters=Parameters(tensors=[], tensor_type=""))
    
    def fit(self, ins: FitIns) -> FitRes:
        """Perform client-side training."""
        # This is handled by NVFlare communication, not used directly
        raise NotImplementedError("NVFlareClientProxy.fit is not implemented")
        #print("IS THIS REALLY CALLED???????????")
        #xxxxxx
        #if self.result:
        #    return self.result
        #return FitRes(parameters=Parameters(tensors=[], tensor_type=""), num_examples=0)
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate model on client data."""
        # This is handled by NVFlare communication, not used directly
        return EvaluateRes(loss=0.0, num_examples=0, metrics={})
    
    def reconnect(self, ins: ReconnectIns) -> None:
        """Handle reconnect instruction."""
        # No-op as reconnection is handled by NVFlare
        pass


class FlowerClientManager(FlwrClientManager):
    """Adapter class that wraps NVFlare's ClientManager to work with Flower's ClientManager interface."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        
    def num_available(self) -> int:
        """Return the number of available clients."""
        return self.num_clients
    
    def register(self, client: ClientProxy) -> bool:
        """This is a no-op as registration is handled by NVFlare."""
        # Registration is handled by NVFlare's ClientManager
        return True
    
    def unregister(self, client: ClientProxy) -> None:
        """This is a no-op as unregistration is handled by NVFlare."""
        # Unregistration is handled by NVFlare's ClientManager
        pass
    
    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients as ClientProxy objects."""
        return [NVFlareClientProxy(cid=f"client-{i}") for i in range(self.num_clients)]   # TODO: use client names from NVFlare
    
    def wait_for(self, num_clients: int, timeout: int) -> bool:
        """Wait until at least `num_clients` are available.
        
        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for.
            
        Returns
        -------
        success : bool
            True if the required number of clients became available before the timeout,
            False otherwise.
        """
        print(f"wait_for: {num_clients}, {timeout}")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if len(self.nvflare_client_manager.clients) >= num_clients:
                return True
            time.sleep(0.5)  # Check every half second
        return False
    
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[object] = None,
    ) -> List[ClientProxy]:
        """Sample a number of available clients."""
        # This will be handled by NVFlare's sampling mechanism
        # We just provide a stub implementation
        print(f"sample: num_clients: {num_clients}, min_num_clients: {min_num_clients}, criterion: {criterion}")
        if num_clients != min_num_clients:
            raise ValueError("num_clients != min_num_clients")

        # TODO: implement criterion
        if criterion is not None:
            raise ValueError("criterion is not supported")

        all_clients = self.all()  # This returns a list already
        return all_clients[:min(num_clients, len(all_clients))]


class FlwrController(BaseFedAvg):
    def __init__(self, strategy: FlwrStrategy, project_name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.project_name = project_name

        # Create mockup of Flower's ClientManager interface
        self.flower_client_manager = FlowerClientManager(self.num_clients)
    
    def update_model_with_aggregated_flwr_parameters(self, model, aggregated_parameters):
        """Update model with aggregated parameters from strategy"""

        print(f"aggregated_parameters.tensor: {type(aggregated_parameters.tensors)}")
        # build state dictionary from aggregated_parameters.tensors
        aggregated_state_dict = {}
        for key, tensor in zip(model.params.keys(), aggregated_parameters.tensors):
            aggregated_state_dict[key] = bytes_to_ndarray(tensor)  # need to convert back to ndarray TODO: avoid this conversion (disable automatic serialization in Flower strategy)

        # convert aggregated_parameters to Flare parameters
        aggregated_model = FLModel(
            params=aggregated_state_dict,
            start_round=model.start_round,
            total_rounds=model.total_rounds,
            current_round=model.current_round,
        )
        model = self.update_model(model, aggregated_model)
        return model
    
    def run(self) -> None:
        self.info("Start FedAvg with Flower Strategy.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            # Configure fit instructions using the strategy with our client manager
            fit_ins = self.strategy.configure_fit(
                server_round=self.current_round,
                parameters=model.params,  # current model parameters
                client_manager=self.flower_client_manager  # Use our adapter
            )
            
            # Sample clients for this round using NVFlare's mechanism 
            # TODO: sample could be used from Flower's ClientManager
            nvflare_clients = self.sample_clients(self.num_clients)
            self.info(f"Sampled {len(nvflare_clients)} clients for round {self.current_round}")

            # Send fit instructions to all clients at once and collect results
            fit_results = []
            # Send model with fit instructions to all clients simultaneously
            results = self.send_model_and_wait(targets=nvflare_clients, data=model)
            
            if results:
                # Process results from all clients
                for result in results:
                    print(f"result: {result}")
                    print(f"result.meta: {result.meta}")
                    client_name = result.meta.get("client_name", f"unknown-{len(fit_results)}")
                    # Convert Flare result to Flower result
                    flower_result = FitRes(
                        # Build tensors from Flare parameters
                        # TODO: avoid serializing the parameters again...
                        parameters=Parameters(tensors=[ndarray_to_bytes(result.params[k]) for k in result.params.keys()], tensor_type="") ,
                        num_examples=result.meta.get("NUM_STEPS_CURRENT_ROUND"),
                        metrics=result.metrics,
                        status=Status(code=Code.OK, message="OK"),
                    )
                    fit_results.append((
                        NVFlareClientProxy(cid=client_name, result=result),
                        flower_result
                    ))

            self.info(f"Collected {len(fit_results)} fit results")

            # Aggregate results using the strategy
            if fit_results:
                aggregate_result = self.strategy.aggregate_fit(
                    server_round=self.current_round,
                    results=fit_results,
                    failures=[]   # TODO: implement failures
                )
                
                if aggregate_result is not None:
                    # Update model with aggregated parameters
                    aggregated_parameters, metrics = aggregate_result
                    model = self.update_model_with_aggregated_flwr_parameters(model, aggregated_parameters)
                    
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
