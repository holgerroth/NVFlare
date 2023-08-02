#!/usr/bin/env python

import numpy as np
import json
import csv
import os
import argparse
import pandas
#pandas.set_option('display.max_columns', None)  # display all columns

MISSING = 'miss'


def load_sizes_json(filename):
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise ValueError('Error: json file could not be read: ' + str(e))
    return data


def load_json(filename):
    with open(filename, 'r') as f:
        # replace non-json characters
        json_str = f.read()  # read the file to string
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace("nan", "null")
        json_str = json_str.replace("NaN", "null")
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise ValueError('Error: json file could not be read: ' + str(e))
    return data


def get_data_list_keys(results):
    data_list_keys = []
    for data_key in results:
        if results[data_key]:
            for model_key in results[data_key]:
                data_lists = list(results[data_key][model_key].keys())
                if len(data_lists) > 0:
                    data_list_keys = data_lists
                    break

    assert len(data_list_keys) > 0, "No data list keys found!"
    return data_list_keys


def get_column_names(results):
    column_names = []
    for data_key in results:
        if results[data_key]:
            columns = list(results[data_key].keys())
            for column in columns:
                if column not in column_names:
                    column_names.append(column)
    return column_names


def make_csv(results, data_list_keys, column_names, output_file_name='results', output_dir='.',
             metrics=[None], summarize=False, sizes=None):
    print('data_list_keys:', data_list_keys)
    if summarize:
        assert len(metrics) == 1, "summarize is only works with one metric!"

    n_missing_total = 0
    n_locals = []
    for data_list_key in data_list_keys:
        print('processing', data_list_key)
        output_name = f"{output_file_name}_{data_list_key}.csv"
        output_file_path = os.path.join(output_dir, output_name)
        #with open(output_file_path, 'w') as f:
        first_column = f"Client [{data_list_key}]"
        header = [first_column]
        header.extend(column_names)
        #writer = csv.DictWriter(f, fieldnames=header)
        df = pandas.DataFrame(columns=header)

        #writer.writeheader()
        n_rows = 0
        n_missing = 0
        for metric in metrics:
            for data_key in column_names:  # add results in same order as column names
                if data_key in results: # only add available results
                    #row = {x: results[data_key][x][data_list_key] for x in results[data_key]}
                    row = {}
                    for x in results[data_key]:
                        if data_list_key in results[data_key][x]:
                            #row[x] = results[data_key][x][data_list_key]
                            row[x] = results[data_key][x]  # no data list
                        else:
                            row[x] = MISSING
                            print('MISSING!')
                            n_missing += 1

                    row[first_column] = data_key
                    if metric:  # only write selected metric
                        new_row = {first_column: row[first_column]+'_'+metric}
                        for entry in row:
                            if entry != first_column:
                                if row[entry] != MISSING:
                                    if isinstance(row[entry][metric], float):
                                        new_row[entry] = row[entry][metric]
                                    else:  # if null in json
                                        new_row[entry] = np.nan
                                else:
                                    new_row[entry] = MISSING
                                    n_missing += 1
                        #writer.writerow(new_row)
                        n_rows += 1
                        df = pandas.concat([df, pandas.DataFrame(new_row, index=[n_rows])])
                    else:
                        #writer.writerow(row)
                        n_rows += 1
                        df = pandas.concat([df, pandas.DataFrame(row, index=[n_rows])])
        print(f'added {n_rows} rows')
        if n_missing > 0:
            print(f'[WARNING] there are {n_missing} missing entries!')
            n_missing_total +=  n_missing

        if summarize:
            # compute local mean
            df_summary = df.copy(deep=True)
            df_summary = df_summary.replace(MISSING, 'NaN')
            values = np.asarray(df_summary)[:, 1::].astype(dtype=np.float)
            diags = np.diag(values)
            n_local = len(diags)
            local_mean = np.nanmean(diags)
            n_locals.append(n_local)
            print(f'Avg. local: {local_mean} (n={n_local})')

            # compute off-diag means
            non_local_means = []
            w_non_local_means = []
            means = []
            n_non_locals = []
            df_means = pandas.DataFrame(columns=header)
            if sizes:
                df_means_weighted = pandas.DataFrame(columns=header)
                df_means_others_weighted = pandas.DataFrame(columns=header)
                total_size = 0
                for d in sizes:
                    total_size += sizes[d]
                w_local_mean = 0.0
                for k, n in enumerate(column_names):
                    if 'server' not in n and ~np.isnan(diags[k]):  # weight by client sizes only
                        assert n in sizes, f"Please provide a dataset size for {n} if weighting by size!"
                        w_local_mean += float(sizes[n]) * diags[k]
                w_local_mean /= float(total_size)
            df_means_others = pandas.DataFrame(columns=header)
            for i, name in enumerate(column_names):
                column = np.asarray(df_summary[name]).astype(dtype=np.float)
                mean = np.nanmean(column)
                n_mean = len(column)
                means.append(mean)
                print(f'mean for {name}: {mean} (n={n_mean})')
                df_means[name] = [mean]
                if sizes:  # weighted mean
                    w_mean = 0.0
                    w_mean_others = 0.0
                    total_size_others = 0
                    for k, n in enumerate(column_names):  # iterated to all values in column by name
                        # local weighted mean
                        if 'server' not in n and ~np.isnan(column[k]):  # weight by client sizes only
                            assert n in sizes, f"Please provide a dataset size for {n} if weighting by size!"
                            w_mean += float(sizes[n])*column[k]
                        # non-local weighted mean
                        if 'server' not in n and ~np.isnan(column[k]) and k != i:  # weight by client sizes only
                            assert n in sizes, f"Please provide a dataset size for {n} if weighting by size!"
                            w_mean_others += float(sizes[n])*column[k]
                            total_size_others += sizes[n]
                    w_mean /= float(total_size)
                    w_mean_others /= float(total_size_others)
                    df_means_weighted[name] = [w_mean]
                if 'server' not in name:  # also add others value if not server model
                    if sizes:
                        df_means_others_weighted[name] = [w_mean_others]
                    # computed unweighted average
                    column = np.delete(column, i)  # remove local value
                    non_local_mean = np.nanmean(column)
                    n_non_local = len(column)
                    non_local_means.append(non_local_mean)
                    if sizes:
                        w_non_local_means.append(w_mean_others)
                    n_non_locals.append(n_non_local)
                    print(f'non-local mean for {name}: {non_local_mean} (n={n_non_local})')
                    df_means_others[name] = [non_local_mean]

            assert len(np.unique(n_non_locals)) == 1, f"number local metrics is not consistent {n_non_locals}!"
            non_local_total_mean = np.nanmean(non_local_means)
            n_non_local_total_mean = len(non_local_means)
            if len(w_non_local_means) > 0:
                w_non_local_total_mean = np.nanmean(w_non_local_means)
            print(f'Avg. others (excl. server): {non_local_total_mean} (n={n_non_local_total_mean})')
            df_means[first_column] = ['Avg. (incl. local)']
            df = df.append(df_means)
            if sizes:
                df_means_weighted[first_column] = ['w. Avg. (incl. local)']
                df = df.append(df_means_weighted)
            df_means_others[first_column] = ['Avg. others']
            df = df.append(df_means_others)
            if sizes:
                df_means_others_weighted[first_column] = ['w. Avg. others']
                df = df.append(df_means_others_weighted)

            # add avg. local
            df_avg = pandas.DataFrame(columns=header)
            df_avg[first_column] = ['Avg. local']
            df_avg[column_names[0]] = [local_mean]
            df = df.append(df_avg)

            # add weighted avg. local
            if sizes:
                df_avg = pandas.DataFrame(columns=header)
                df_avg[first_column] = ['w. Avg. local']
                df_avg[column_names[0]] = [w_local_mean]
                df = df.append(df_avg)

            # add avg. others
            df_avg = pandas.DataFrame(columns=header)
            df_avg[first_column] = ['Avg. others (excl. server)']
            df_avg[column_names[0]] = [non_local_total_mean]
            df = df.append(df_avg)

            # add weighted avg. local
            if sizes:
                df_avg = pandas.DataFrame(columns=header)
                df_avg[first_column] = ['w. Avg. others (excl. server)']
                df_avg[column_names[0]] = [w_non_local_total_mean]
                df = df.append(df_avg)

        print(df)
        df.to_csv(output_file_path)
        print(f'saved {data_list_key} results in {output_file_path}')
    
    if n_missing_total > 0:
        print(f'[WARNING] there are a total {n_missing_total} missing entries for {np.unique(n_locals)} clients!')
    else:
        print(f'complete (no missing values for {np.unique(n_locals)} clients!).')


def convert_json_to_csv(filename, output_dir, output_file_name, metrics=[None], summarize=False, weight_by_size=None):
    results = load_json(filename)
    if weight_by_size:
        sizes = load_sizes_json(weight_by_size)
    else:
        sizes = None

    column_names = get_column_names(results)
    column_names.sort()

    data_list_keys = get_data_list_keys(results)
    # move server results to end
    for name in ['server', 'server_best_model']:
        if name in column_names:
            column_names.remove(name)
            column_names.append(name)

    if output_file_name is None:  # use json filename without extension
        output_file_name = os.path.splitext(filename)[0] + '_' + '_'.join(metrics)

    assert isinstance(metrics, list), f'metrics is expected to be a list but is {metrics}'
    make_csv(results=results, data_list_keys=data_list_keys, column_names=column_names,
             output_dir=output_dir, output_file_name=output_file_name, metrics=metrics,
             summarize=summarize, sizes=sizes)


METRICS_AUC='auc_1,auc_2,auc_3,auc_4,auc_avg'
METRICS_AUC='auc_1,auc_2,auc_3'
METRICS_AUC='auc_avg'
METRICS_F1='f1_1,f1_2,f1_3,f1_4,f1_avg'
METRICS_BALANCED_ACCURACY='balanced_accuracy_1,balanced_accuracy_2,balanced_accuracy_3,balanced_accuracy_4,balanced_accuracy_avg'
METRICS_ACCURACY='accuracy_1,accuracy_2,accuracy_3,accuracy_4,accuracy_avg'
METRICS_DICE="mean_dice_class2"
METRICS_COVID_XRAY="val_acc"
METRICS="accuracy"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make csvs from cross val results json.")
    parser.add_argument('-n', '--name', type=str, help='Path or name of json file.', required=True)
    parser.add_argument('-d', '--dir', type=str, default='.', help='Path to output dir')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    parser.add_argument('-m', '--metrics', type=str, default=METRICS, help='List of metrics')
    parser.add_argument('-w', '--weight_by_size', type=str, default=None, help='json file with dataset sizes')
    parser.add_argument('-s', '--summarize', action='store_true', help='Compute summary metrics '
                              '(only works if only one metric is specified)')

    args = parser.parse_args()

    if args.metrics:
        metrics = args.metrics.split(',')
    else:
        metrics = [None]

    convert_json_to_csv(args.name, args.dir, args.output, metrics=metrics, summarize=args.summarize,
                        weight_by_size=args.weight_by_size)
