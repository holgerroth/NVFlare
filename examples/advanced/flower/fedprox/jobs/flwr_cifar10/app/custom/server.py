import flwr as fl

# Start Flower server
print("Running Server code...")
fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=fl.server.ServerConfig(num_rounds=3),
)
