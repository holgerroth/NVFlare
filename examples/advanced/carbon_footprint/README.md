# Carbon Footprint Example with NVFlare

This example demonstrates how to measure the carbon footprint of federated learning using [CodeCarbon](CodeCarbon.io) in offline mode.

## Prerequisites

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

3. Install NVFlare and required dependencies:
```bash
pip install -r requirements.txt
pip install pandas matplotlib seaborn
```

## Running the Example

1. Run the example:
```bash
python job.py
```

## Understanding the Output

The example will:
1. Create a federated learning job with 2 clients
2. Run for 2 rounds
3. Measure and report carbon emissions in offline mode

The carbon emissions data will be saved in a CSV file in the current directory, typically named `emissions.csv` under each clients workdir in `/tmp/nvflare/carbon_footprint`. 

You can check the rsult using, e.g. 

```bash
cat /tmp/nvflare/carbon_footprint/site-0/emissions.csv
```

## Plotting the Results

To visualize the carbon emissions data from all clients:

1. Run the plotting script:
```bash
python plot_emissions.py --base_dir="/tmp/nvflare/carbon_footprint"
```

This will generate a plot showing the cumulative carbon emissions over time for each client. The plot will be saved as `carbon_emissions_plot.png` in the current directory.

The plot includes:
- Cumulative emissions over time for each client
- Clear labels and legend
- Timestamps on the x-axis
- Emissions in kg CO2 on the y-axis