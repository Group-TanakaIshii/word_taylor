# -*- coding: utf-8 -*-
# Available with Python 3.6 or later version.
import argparse
import pandas as pd
import numpy as np
import pickle
import time
import re
from matplotlib import pyplot as plt
from statistics import mean, stdev
from scipy import optimize
from os.path import expanduser, exists


def input_cmdln():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", metavar="FilePath", type=str, nargs=1,
                        help="input file path to read")
    parser.add_argument("-w", "--width", metavar="Width", type=int, nargs=1,
                        help="input width of division (default:5620)")
    parser.add_argument("--nodraw", action="store_true", default=False,
                        help="don't output figure if this flag is set (default: False)")
    parser.add_argument("-p", "--pickle", action="store_true", default=False,
                        help="use cached pickle file")
    args = parser.parse_args()
    rslt = {"filepath": args.filepath[0]}
    rslt["nodraw"] = args.nodraw
    rslt["pickle"] = args.pickle
    if args.width:
        rslt["width"] = args.width[0]
    return rslt


def return_wordloc(word_set, data):
    wordloc = {}
    for i in range(len(data)):
        if data[i] in word_set:
            if data[i] not in wordloc:
                wordloc[data[i]] = [i]
            else:
                wordloc[data[i]].append(i)
    return wordloc


def basic_process(word_data):
    text = re.sub("[^一-龥ぁ-んァ-ン\w０-９\n]",
                  "",
                  word_data)
    text = re.sub(r'(\n)+', "\n", text)
    return text.replace("\u3000\n", "").split("\n")


def load_textdata(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def load_pickle(data_path, pickle_flag=False):
    fname = ".".join(data_path.split("/")[-1].split(".")[:-1])
    if False: ### Choosing 2000 words: stop using ###
        if not exists(pickle_path):
            data = dl.basic_process(dl.load_textdata(data_path))
            length = len(data)
            word_set = set(data[int(length/2):int(length/2) + 2000])
            wordloc = return_wordloc(word_set, data)
            with open(pickle_path, mode="wb") as f:
                pickle.dump([wordloc, length], f)
        else:
            with open(pickle_path, mode="rb") as f:
                wordloc, length = pickle.load(f)
    else: ### Choosing all words: on trial ###
        data = basic_process(load_textdata(data_path))
        length = len(data)
        word_set = set(data)
        wordloc = return_wordloc(word_set, data)
    return wordloc, length, fname


def TFS(loc, width, length):
    nums = [0] * int(length / width)
    for appear in loc:
        if appear / width < int(length / width):
            nums[int(appear / width)] += 1
    return [mean(nums), stdev(nums)]


def residual_func(parameter, x, y):
    residual = np.log10(y) - np.log10(parameter[0] * x ** parameter[1])
    return residual


def log_leastsq_regression(xdata, ydata):
    param1 = np.array([1.0, 0.0])
    result = optimize.leastsq(residual_func,
                              x0=param1,
                              args=(list(xdata),
                                    list(ydata)))
    param = result[0]
    average_err = np.sqrt(mean(np.power(np.array(residual_func(param, list(xdata), list(ydata))), 2)))
    return param, average_err


def plot_TFS(data_points, fname, width, plot=True):
    print("Regression ...", end="")
    optim_param, average_err = log_leastsq_regression(data_points["mu"], data_points["sigma"])
    print("Done!")
    alpha = optim_param[1]
    if plot:
        data_points["reg"] = optim_param[0] * data_points["mu"] ** optim_param[1]
        ax = data_points.plot(kind="scatter", x="mu", y="sigma", color="royalblue")
        data_points.plot(x="mu",y="reg", ax=ax, color="navy", legend=False)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("μ", fontsize=26)
        plt.ylabel("σ", fontsize=26)
        plt.tick_params(labelsize = 26)
        plt.xlim([data_points["mu"].min(), data_points["mu"].max()])
        plt.ylim([data_points["sigma"].min(), data_points["sigma"].max()])
        plt.text(data_points["mu"].max() ** (8/15) * data_points["mu"].min() ** (7/15),
                 data_points["sigma"].max() ** (1/15) * data_points["sigma"].min() ** (14/15),
                 f"α={'%.3f' % alpha}\nε={'%.3f' %average_err}",
                 size=26)
        # plt.title(f"TFS on {fname} (w={str(width)})")
        # plt.savefig(f"{fname}_w={str(width)}.png")
        plt.show()
    return alpha


def main():
    # data load
    cmdln = input_cmdln()
    print("Loading data...", end="")
    before_load = time.time()
    wordloc, length, fname = load_pickle(cmdln["filepath"], cmdln["pickle"])
    print(f"Done! Load time: {time.time() - before_load}s")
    if "width" in cmdln:
        width = cmdln["width"]
    else:
        width = 5620
    # generate plotting points
    print("Counting words ...")
    data_points = pd.DataFrame([], columns=["mu", "sigma"])
    count = 0
    num = len(wordloc)
    for w, loc in wordloc.items():
        count += 1
        data_points = data_points.append(pd.DataFrame([TFS(loc, width, length)], columns=["mu", "sigma"], index=[w]))
        if count % 10000 == 0:
            print(f"{count} / {num} ({100*count/num} %)")
    print("Done!")
    data_points = data_points[data_points.sigma > 0]
    return fname, width, plot_TFS(data_points, fname, width, plot=not cmdln["nodraw"])


if __name__ == "__main__":
    fname, width, alpha = main()
    print(fname, width, alpha)
