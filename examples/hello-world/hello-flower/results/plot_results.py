import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

DPI = 300

flwr_in_flare_files = ["./flwr_in_flare_client1.csv", "./flwr_in_flare_client2.csv"]
flwr_alone_files = ["./flwr_alone_client1.csv", "./flwr_alone_client2.csv"]

headers = ["Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"]
clients = ["Client 1", "Client 2"]


def read_files(files, names, tag, df=None):
    dfs = []
    for file, name in zip(files, names):
        _df = pd.read_csv(file, names=headers)
        _df["Setting"] = len(_df) * [tag]
        _df["Client"] = len(_df) * [name]
        _df["Round"] = [i for i in range(len(_df))]
        dfs.append(_df)
    new_df = pd.concat(dfs)

    if df is not None:
        new_df = pd.concat([df, new_df])

    return new_df


df_in_flare = read_files(flwr_in_flare_files, names=clients, tag="Flower in NVFlare")
df_alone = read_files(flwr_alone_files, names=clients, tag="Flower Alone")

# plot
#plt.subplot(1, 2, 1)
plt.figure()
sns.lineplot(data=df_alone, x="Round", y="Train Loss", hue="Client")
#plt.title("Flower Alone")
#plt.subplot(1, 2, 2)
plt.savefig("flwr_alone.pdf", dpi=DPI)

plt.figure()
sns.lineplot(data=df_in_flare, x="Round", y="Train Loss", hue="Client")
#plt.title("Flower in NVFlare")
plt.savefig("flwr_in_flare.pdf", dpi=DPI)
#plt.show()
