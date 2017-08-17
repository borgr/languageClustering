
# coding: utf-8

# In[1]:
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/eurasian-phonologies/src/")
import platform
from subprocess import call
import subprocess
import numpy as np
import pandas as pd
from random import randint
from munkres import Munkres
munkres = Munkres()
from collections import Counter
from itertools import product
import re
from functools import reduce
import json
import pickle
import inspect
# print (inspect.stack()[0][3])
import metric_learn as ml
from IPAParser import parsePhon
import scipy.spatial.distance as spdist
from sklearn import linear_model
from sklearn.feature_selection import RFECV, RFE
# import rpy2.robjects as robjects

# from rpy2.robjects import r, pandas2ri
NAME = "name"
GROUP ='group'
LAT = 'lat'
LON ='lon'
INV = "inv"
PARSED = "parsed"
GRAM_NAME = "gram"
FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def escape_print(s): 
    print(s.encode('unicode_escape'))


def process_phoneme(p):
    p = p.split('/')[0].replace(
        ':', 'ː').replace(
        '\u0361', '').replace(
            'ˠ', 'ˤ').replace(
                '\u033b', '').replace(
                    "'", 'ʰ').replace(
                        '\u032a', '')
    if 'l' not in p and '\u0334' in p:
        p = p.replace('\u0334', 'ˤ')
    status = 'normal'
    if '(' in p:
        status = 'borrowed'
    elif '<' in p:
        status = 'marginal'
    return p.strip('()<>'), status


def process_row(r):
    phons = []
    status = []
    for el in re.split(r'\s+', r[1]['Cons'].strip()):
        p, s = process_phoneme(el)
        if not p:
            print(el)
            print(r[1]['Cons'])
            print(r[1]['Name'])
            print()
        phons.append(p)
        status.append(s)
    return pd.DataFrame({
        'Name': np.repeat(r[1]['Name'], len(phons)),
        'Group': np.repeat(r[1]['Group'], len(phons)),
        'Lat': np.repeat(r[1]['Lat'], len(phons)),
        'Lon': np.repeat(r[1]['Lon'], len(phons)),
        'Cons': phons,
        'Status': status
    })


def binarise_lang(r, all_phons):
    items = [
        (NAME, np.repeat(r['Name'].iloc[0], 1)),
        (GROUP, np.repeat(r['Group'].iloc[0], 1)),
        (LAT, np.repeat(r['Lat'].iloc[0], 1)),
        (LON, np.repeat(r['Lon'].iloc[0], 1))
    ]
    local_df = pd.DataFrame.from_items(items)
    return binarize_db(local_df, r['Cons'], all_phons)


def binarise_euro_lang(r, all_phons, key):
    items = [
        # (NAME, [r[NAME]]),
        (NAME, [key]),
        (GROUP, [r['gen'][0]]),
        (LAT, [r['coords'][0]]),
        (LON, [r['coords'][1]])]
    local_df = pd.DataFrame.from_items(items)
    return binarize_db(local_df, r['cons'], all_phons)


def binarize_db(db, current, all):
    current = set(current)
    for elem in all:
        db[elem] = [int(elem in current)]
    return db


def read_kurd():
    # In[5]:

    data = pd.read_csv('june_6_data.tsv', sep='\t')
    dataframe = reduce(lambda x, y: x.append(process_row(y),
                                             ignore_index=True),
                       data.iterrows(),
                       pd.DataFrame())

    return dataframe


def read_euro():
    """" returns a dict of dicts  with 
    names:['type', 'vows', 'code',
           'cluster', 'cons', 'name',
           'tones', 'coords', 'source',
           'gen', 'inv', 'comment',
           'contr', 'finals', 'syllab']
           """
    db_file = "eurasian-phonologies/src/dbase/phono_dbase.json"
    with open(db_file, "r") as fl:
        data = json.load(fl) 
    return data


def parseInv(inv):
    return [parsePhon(x) for x in inv]


def parse(inv_file="", feature_file="", base_file=""):
    if inv_file and feature_file and base_file:
        if os.path.isfile(inv_file) and os.path.isfile(feature_file) and os.path.isfile(base_file):
            print("reading dbs from file")
            return pd.read_pickle(inv_file), pd.read_pickle(feature_file), pd.read_pickle(base_file)
    dataframe = read_kurd()

    # collect all phonemes
    euro_db = read_euro()
    all_phons = set(dataframe['Cons'].unique())
    for key in euro_db:
        all_phons.update(set(euro_db[key]["cons"]))

    # construct kurd db
    binary_df_pure = pd.DataFrame()
    # """Note, this is not an efficient way to create a db, 
    # better to save as list of series\DF and then merge or create a new one of lists"""
    binary_df_w_borrowings = pd.DataFrame() 
    for l in dataframe['Name'].unique():
        binary_df_pure = binary_df_pure.append(binarise_lang(dataframe[(
            dataframe['Name'] == l) & (dataframe['Status'] != 'borrowed')], all_phons))
        # binary_df_w_borrowings = binary_df_w_borrowings.append(
        #     binarise_lang(dataframe[dataframe['Name'] == l], all_phons))

    # create a db containing all data
    base_db = []
    for key in binary_df_pure[NAME].unique():
        row = binary_df_pure[binary_df_pure[NAME] == key]
        inv = [name for name in all_phons if int(row[name]) != 0]
        parsed = parseInv(set(inv))
        base_db.append((key, row.get_value(0, LON), row.get_value(0, LAT), row.get_value(0, GROUP), inv, parsed))

    # create a db containing binarized inventories 
    inv_db = binary_df_pure.copy()
    for key in euro_db:
        row = euro_db[key]
        inv = row[INV]
        inv_db = inv_db.append(binarise_euro_lang(row, all_phons, key))
        base_db.append((key, row["coords"][0], row["coords"][1], row["gen"][0], inv, parsed))
    
    # binarize parsed
    base_db = pd.DataFrame(base_db, columns=[NAME, LON, LAT, GROUP, INV, PARSED])

    base_db.set_index(NAME, drop=False, inplace=True)
    inv_db.set_index(NAME, drop=False, inplace=True)

    all_pre = set()
    all_features = set()
    all_post = set()
    for i, row in base_db.iterrows():
        for parsed in row[PARSED]:
            all_pre.update(set(parsed[0]))
            all_features.update(set(parsed[1]))
            all_post.update(set(parsed[2]))
    all_features = flatten_parse([all_pre, all_features, all_post])

    # create a db containing binarized features
    feature_db = pd.DataFrame()
    for i, r in base_db.iterrows():
        items = [
            (NAME, [r[NAME]]),
            (GROUP, [r[GROUP]]),
            (LAT, [r[LAT]]),
            (LON, [r[LON]])]
        local_df = pd.DataFrame.from_items(items)
        feature_db = feature_db.append(binarize_db(local_df, [flatten_parse(x) for x in r[PARSED]], all_features))

    feature_db.set_index(NAME, drop=False, inplace=True)
    if inv_file and feature_file and base_file:
        print("dumping to ", inv_file)
        inv_db.to_pickle(inv_file) 
        print("dumping to ", feature_file)
        feature_db.to_pickle(feature_file) 
        print("dumping to ", base_file)
        base_db.to_pickle(base_file)
    return inv_db, feature_db, base_db


def anounce_finish():
    if sys.platform == "linux":
        if set(("debian", "Ubuntu")) & set(platform.linux_distribution()):
            #perhaps works only in ubuntu?
            subprocess.call(['speech-dispatcher'])        #start speech dispatcher
            subprocess.call(['spd-say', '"your process has finished"'])
        else:
            a = subprocess.Popen(('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 300, 2)).split())
    elif sys.platform == "darwin":
        subprocess.call('say "your program has finished"'.split())
    else:
        import winsound
        winsound.Beep(300,2)


def flatten_parse(parse):
    content = set(parse[1])
    content.update(["pre_" + x for x in parse[0]])
    content.update(["post_" + x for x in parse[2]])
    return frozenset(content)


def to_comparable_array(row, names=False):
    """ gets a row of inventory db and returns only the inventory"""
    if names:
        return row.index.values[~row.index.isin([NAME,LAT,LON,GROUP])]
    return np.array(row[~row.index.isin([NAME,LAT,LON,GROUP])], np.float64)


def create_col_remover(indexes):
    if type(indexes) != type([]):
        indexes = [indexes]
    def remover(row):
        return np.array(row[~row.index.isin(indexes)])
    return remover


def get_parsed(row):
    """creates a list of frozen sets from a base db row"""
    res = []
    for item in row[PARSED]:
        res.append(flatten_parse(item))
    return res


def bag_of_features_getter(n):
    def get_ngrams(row, names=False):
        res = []
        for key in row.index.values:
            if key.startswith(str(n) + GRAM_NAME):
                if names:
                    res.append(key[len(str(n)+GRAM_NAME):])
                else:
                    res.append(row[key])
        return res
    return get_ngrams


def counting_feature_dist(feature_set_a, feature_set_b):
    return (sum([1 for a in feature_set_a if a not in feature_set_b]) + 
            sum([1 for b in feature_set_b if b not in feature_set_a])) / (
            len(feature_set_a) + len(feature_set_b))


def aligning_dist(a, b, dist=counting_feature_dist):
    """ sums the distances of the most closely related phonemes""" 
    empty_feature = frozenset()
    a = list(a)
    b = list(b)

    assert(dist(a[0], frozenset())) == 1
    #find lengths
    length_dif = len(a) - len(b)
    if length_dif > 0:
        shorter = b
        longer = a
        switched = False
    else:
        shorter = a
        longer = b
        switched = True
        length_dif = abs(length_dif)
    shorter += [empty_feature] * length_dif

    #create matrix  
    matrix = np.zeros((len(longer), len(longer)))
    for i in range(len(longer)):
        for j in range(len(longer) - length_dif):
            matrix[i,j] = dist(longer[i], shorter[j])
    
    #compare with munkres
    indexes = munkres.compute(matrix)
    return sum([dist(a[idx_a], b[idx_b]) for idx_a, idx_b in indexes])


def bag_of_features_dist(a, b, dist=spdist.cosine):
    """counts the number of occurences for each feature, regardless of in which phoneme it is"""
    count_a = Counter()
    count_b = Counter()
    for item in a:
        count_a.update(item)
    for item in b:
        count_b.update(item)
    list_a = []
    list_b = []
    for item in set(list(count_a.keys()) + list(count_b.keys())):
        list_a.append(count_a[item])
        list_b.append(count_b[item])
   
    res = dist(list_a, list_b)
    assert(not (res == 0 and list_a != list_b))
    return res


def bag_of_ngram_features_dist(a, b, n=2, dist=spdist.cosine):
    """counts the number of occurences for each feature, regardless of in which phoneme it is"""
    count_a = Counter()
    count_b = Counter()
    for item in a:
        count_a.update(item)
    for item in b:
        count_b.update(item)
    list_a = []
    list_b = []
    unigrams = set(list(count_a.keys()) + list(count_b.keys()))
    for item in product(unigrams, repeat=n):
        item = set(item)
        list_a.append(sum([1 for phoneme in a if len(phoneme & item) == n]))
        list_b.append(sum([1 for phoneme in b if len(phoneme & item) == n]))
    res = dist(list_a, list_b)
    assert(not (res == 0 and list_a != list_b))
    return res


def gencoordinates(m, n):
    seen = set()
    x, y = randint(m, n), randint(m, n)
    while len(seen) < (n + 1 - m)**2:
        while (x, y) in seen:
            x, y = randint(m, n), randint(m, n)
        seen.add((x, y))
        yield (x, y)
    return


def genidx(m, n):
    seen = set()
    x = randint(m, n)
    while len(seen) < (n + 1 - m):
        while x in seen:
            x = randint(m, n)
        seen.add(x)
        yield x
    return


# def extract_inv(db, train_percentage):
#     x = []
#     y = []
#     for i, row in zip(range(train_percentage * db.shape[0] / 100, genidx(0, db.shape[0] - 1))):
#         x.append()
#     # train_size = (db.shape[0] * (db.shape[0] - 1))/2 * train_percentage / 100 # n*n-1/2 as it is a symmetric matrix
#     # generator = gencoordinates(0,db.shape[0] - 1)
#     # for i,(row, col) in range(zip(range(train_size), generator)):
#         # y.append(db.iloc[row][GROUP] == db.iloc[col][GROUP])
        # x.append()


def learn_metric_by_diffs(db, extract, learning_algorithm, train_percentage=10):
    x = []
    y = []
    As = []
    Bs = []
    Cs = []
    Ds = []
    groups = []
    subset = list(zip(range(int(train_percentage * db.shape[0] / 100)), genidx(0, db.shape[0] - 1)))
    for _, row in subset:
        x.append(extract(db.iloc[row]))
        groups.append(db.iloc[row][GROUP])
    for i_a, a in enumerate(groups):
        for i_b, b in enumerate(groups):
            if i_a == i_b:
                continue
            for i_c, c in enumerate(groups):
                for i_d, d in enumerate(groups):
                    if i_c == i_d:
                        continue
        if (a == b and c != d):
            As.append(i_a)
            Bs.append(i_b)
            Cs.append(i_c)
            Ds.append(i_d)

    return learning_algorithm.fit(np.array(x), (As, Bs, Cs, Ds))


def train_model(db, extract, learning_algorithm, train_percentage=10, save=""):
    if save and os.path.isfile(save):
        print("reading", save)
        with open(save, "rb") as fl:
            return pickle.load(fl)
    x = []
    y = []
    names = []
    for i, row in zip(range(int(train_percentage * db.shape[0] / 100)), genidx(0, db.shape[0] - 1)):
        x.append(extract(db.iloc[row]))
        # print(x[-1])
        cur_name = db.iloc[row][GROUP]
        # find place without iterating twice on the list
        found = False
        for i, name in enumerate(names):
            if name == cur_name:
                y.append(i)
                found = True
                break
        if not found:
            y.append(len(names))
            names.append(cur_name)
    model = learning_algorithm.fit(np.array(x), np.array(y))
    if save:
        with open(save, "wb") as fl:
            pickle.dump(model, fl)
    return model


def to_numerical_classes(srs):
    contents = []
    origin = srs.copy()
    origin.name = "names"
    for idx, (name, cur_content) in enumerate(srs.iteritems()):
        found = False
        for i, content in enumerate(contents):
            if content == cur_content:
                srs[name] = i + 1
                found = True
        if not found:
            srs[name] = len(contents) + 1
            contents.append(cur_content)
    return pd.concat([origin, srs], axis=1)


def create_distance_matrix(db, func, normalize=lambda x:x, save=""):
    """ gets a db and a function and calculate the distance metrics"""
    if save and os.path.isfile(save):
        print("reading", save)
        if save.endswith("csv"):
            res = pd.read_csv(save)
        if save.endswith("pkl") or save.endswith("pckl") or  save.endswith("pcl"):
            res = pd.read_pickle(save)
        if save.endswith("json"):
            res = pd.read_json(save)  
        return res 

    print("calculating distance matrix")   
    d = {}
    sort_by = np.argsort(db[GROUP])
    db = db.iloc[sort_by, :]
    cache_normalize = {}
    for i, r in db.iterrows():
        if i not in cache_normalize:
            cache_normalize[i] = normalize(r)
        index = []
        distances = []
        for j, c in db.iterrows():
            if j not in cache_normalize:
                cache_normalize[j] = normalize(c)
            try:
                same_normalization = all(cache_normalize[i] == cache_normalize[j])
            except:
                same_normalization = cache_normalize[i] == cache_normalize[j]
            if i != j and same_normalization:
                print("for languages", r[NAME], c[NAME],"checked vallues are the same")#, cache_normalize[i], cache_normalize[j])
            index.append(c[NAME])
            distances.append(func(cache_normalize[i], cache_normalize[j]))
        d[r[NAME]] = pd.Series(distances, index=index)
    res = pd.DataFrame(d)[index]
    if save:
        path, basename = os.path.split(save) 
        cluster_filename = path + os.sep + "../clusters/" + basename   
        if save.endswith("csv"):
            res.to_csv(save)
            to_numerical_classes(db[GROUP]).to_csv(cluster_filename, header=True)
            print(cluster_filename)
        elif save.endswith("pkl") or save.endswith("pckl") or  save.endswith("pcl"):
            res.to_pickle(save)
            to_numerical_classes(db[GROUP]).to_pickle(cluster_filename, header=True)
        elif save.endswith("json"):
            res.to_json(save)
            to_numerical_classes(db[GROUP]).to_json(cluster_filename, header=True)
        else:
            print("unknown file extension")
    return res


def create_ngrams(db, n):
    unigrams = Counter()
    for i, row in db.iterrows():
        for item in get_parsed(row):
            unigrams.update(item)
    ngrams = list(product(unigrams, repeat=n))
    names = [str(n)+GRAM_NAME+str(gram)[1:-1].replace("'","") for gram in ngrams]
    res = []
    for i, row in db.iterrows():
        lst = [i]
        parsed = get_parsed(row)
        for name, item in zip(names, ngrams):
            item = set(item)
            lst.append(len([1 for phoneme in parsed if len(phoneme & item) == n]))
        res.append(lst)
    res = pd.DataFrame(res, columns=[NAME] + names)
    res.set_index(NAME, drop=False, inplace=True)
    res = pd.concat([res,db.loc[:,[NAME, GROUP]]], axis=1)
    return res


def create_learned_distance_matrix(db, learning_algorithm, train_percentage, normalize=lambda x:x, save="", save_model=""):
    model = train_model(db, normalize, learning_algorithm, train_percentage, save_model)
    print("calculated model ", save_model)
    comparable_db = np.array([normalize(row) for i, (name, row) in enumerate(db.iterrows())])
    frame = pd.DataFrame(model.transform(comparable_db), index=db.index.values)
    frame[[GROUP,NAME]] = db.loc[frame.index.values].loc[:, [GROUP, NAME]].astype(str)
    create_distance_matrix(frame, spdist.euclidean, create_col_remover([GROUP,NAME]), save)


def find_best_features(db, learning_algorithm, normalize=lambda x:x, train_percentage=100, save="", save_model="", cross_validate=False):
    if cross_validate:
        model = train_model(db, normalize, RFECV(learning_algorithm), train_percentage, save_model)
        sort = np.argsort(model.ranking_)
        print("calculated model ", save_model)
        print("feature names ordered by usefulness:", list(np.array(normalize(db.iloc[1,:], True))[sort]))
        print("features chosen:", list(model.support_[sort]))
        print("feature ranking:", model.ranking_[sort])
        # print("feature names:", normalize(db.iloc[1,:], True))
        # print("features chosen:", model.support_)
    else:
        model = train_model(db, normalize, RFE(learning_algorithm, 1), train_percentage, save_model)
        sort = np.argsort(model.ranking_)
        print("calculated model ", save_model)
        print("feature names ordered by usefulness:", list(np.array(normalize(db.iloc[1,:], True))[sort]))
        # print("features chosen:", list(model.support_[sort]))


def calculate_distances_main(inv_db, feature_db, base_db):
    dist_dir = FILE_PATH + "/distance_metrics/"
    if not os.path.isdir(dist_dir):
        os.mkdir(dist_dir)
    clust_dir = FILE_PATH + "/clusters/"
    if not os.path.isdir(clust_dir):
        os.mkdir(clust_dir)
    models_dir = FILE_PATH + "/models/"
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    train_percentage = 50

    #### run on small db ###
    # print(create_distance_matrix(inv_db.iloc[:14,:], spdist.hamming, to_comparable_array, dist_dir + "inv_hamming.csv"))

    """ metric learning """
    # number_of_features = len(to_comparable_array((next(inv_db.iterrows())[1])))
    # prior = np.random.rand(number_of_features, number_of_features)
    # prior = np.dot(prior, prior.transpose())
    # save_lsml = "lsml_inv"
    # lsml = create_learned_distance_matrix(inv_db, ml.LSML_Supervised(prior=prior), train_percentage, to_comparable_array, clust_dir + save_lsml, models_dir + save_lsml)

    # save_sdml = "sdml_inv"
    # sdml = create_learned_distance_matrix(inv_db, ml.SDML_Supervised(), train_percentage, to_comparable_array, clust_dir + save_sdml, models_dir + save_sdml)

    # save_lmnn = "lmnn_inv"
    # lmnn = create_learned_distance_matrix(inv_db, ml.LMNN(k=5, learn_rate=1e-6), train_percentage, to_comparable_array, clust_dir + save_lmnn, models_dir + save_lmnn)

    # save_nca = "nca_inv"
    # nca = create_learned_distance_matrix(inv_db, ml.NCA(), train_percentage, to_comparable_array, clust_dir + save_nca, models_dir + save_nca)

    save_rca = "rca_inv.csv"
    rca = create_learned_distance_matrix(inv_db, ml.RCA_Supervised(), train_percentage, to_comparable_array, clust_dir + save_rca, models_dir + save_rca)

    save_itml = "itml_inv.csv"
    itml = create_learned_distance_matrix(inv_db, ml.ITML_Supervised(), train_percentage, to_comparable_array, clust_dir + save_itml, models_dir + save_itml)

    """ hand made distances """
    create_distance_matrix(base_db, lambda x,y:bag_of_ngram_features_dist(x, y, 2, spdist.euclidean), get_parsed, dist_dir + "bigram_euclidean.csv")
    create_distance_matrix(inv_db, spdist.yule, to_comparable_array, dist_dir + "inv_yole.csv")
    create_distance_matrix(inv_db, spdist.jaccard, to_comparable_array, dist_dir + "inv_jaccard.csv")
    create_distance_matrix(inv_db, spdist.hamming, to_comparable_array, dist_dir + "inv_hamming.csv")
    create_distance_matrix(base_db, bag_of_ngram_features_dist, get_parsed, dist_dir + "bigram_cosine.csv")
    create_distance_matrix(base_db, bag_of_features_dist, get_parsed, dist_dir + "bag_cosine.csv")
    create_distance_matrix(base_db, lambda x,y:bag_of_features_dist(x,y,spdist.euclidean), get_parsed, dist_dir + "bag_euclidean.csv")
    create_distance_matrix(base_db, aligning_dist, get_parsed, dist_dir + "aligned.csv")

    anounce_finish()


def choose_features_main(inv_db, feature_db, base_db):
    estimator = linear_model.LogisticRegression()
    # find_best_features(base_db, estimator, get_parsed)
    gram_db = create_ngrams(base_db, 1)
    find_best_features(gram_db, estimator, bag_of_features_getter(1))
    gram_db = create_ngrams(base_db, 2)
    find_best_features(gram_db, estimator, bag_of_features_getter(2))
    find_best_features(inv_db, estimator, to_comparable_array)


def clean_rare(db, col, mincount=5):
    counts = db[col].value_counts()
    return db[db[col].isin(counts[counts >= mincount].index.values)]


def main():
    path = FILE_PATH + os.sep + "cache" + os.sep
    if not os.path.isdir(path):
        os.mkdir(path)
    # initialize directories
    inv_file = path + "inv.pckl"
    feature_file = path + "features.pckl"
    base_file = path + "base.pckl"
    inv_db, feature_db, base_db = parse(inv_file, feature_file, base_file)
    inv_db = clean_rare(inv_db, GROUP)
    base_db = clean_rare(base_db, GROUP)
    
    assert(all([x in base_db.index.values for x in inv_db.index.values]))
    assert(all([x in inv_db.index.values for x in base_db.index.values]))

    calculate_distances_main(inv_db, feature_db, base_db)
    choose_features_main(inv_db, feature_db, base_db)

if __name__ == '__main__':
    main()