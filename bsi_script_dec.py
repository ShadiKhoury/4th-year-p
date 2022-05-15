from abc import ABC, abstractmethod
import json
import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Tuple, Union, Callable, Optional, Set, Sequence, List, Dict
from pandas import DataFrame, Series
from numpy import ndarray
import lightgbm as lgb
from sklearn.base import is_classifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import check_scoring, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from multiprocessing import Pool
from tqdm.auto import tqdm
import numpy as np

# region utils

MultiVariateArray = Union[DataFrame, Series, ndarray]
UniVariateArray = Union[Series, ndarray, list]
run_type = "long" # #poc1 #poc2 # long

def replot():
    plt.cla()
    plt.clf()
    plt.close()
    plt.figure()

def context_to_key(context: Iterable[str]) -> Tuple[str]:
    return tuple(sorted(context))


def is_empty(array: MultiVariateArray):
    if isinstance(array, DataFrame) or isinstance(array, Series):
        return array.empty
    else:
        return len(array) == 0


def multi_process_lst(lst, apply_on_chunk, chunk_size=1000, n_processes=1, args=None):
    '''
    applies apply_on_chunk on lst using n_processes each gets chunk_size items from lst each time
    '''
    chunks = split(lst, n_processes)
    chunks = flatten_iterable(group(c, chunk_size) for c in chunks if len(c) > 0)
    if n_processes == 1:
        for c in lst:
            yield apply_on_chunk([c], *args)
    else:
        with Pool(n_processes) as pool:

            for preproc_inst in pool.imap_unordered(unpack_args_wrapper,
                                                    [(apply_on_chunk, (c, *args)) for c in chunks]):
                yield preproc_inst


def unpack_args_wrapper(function_args_tup):
    return function_args_tup[0](*function_args_tup[1])


def flatten_iterable(listoflists):
    return [item for sublist in listoflists for item in sublist]


def split(lst, n_groups):
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def group(lst, max_group_size):
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst) + max_group_size - 1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

# endregion

# region evaluators


EvaluationFunction = Callable[[MultiVariateArray, UniVariateArray,
                               Optional[MultiVariateArray], Optional[UniVariateArray]], float]


class SklearnEvaluator:

    """Creates evaluation function from any SKlearn model"""

    def __init__(self,
                 model,
                 scoring: Optional[str] = None,
                 prior_strategy: Optional[str] = None,
                 cv: int = 3,
                 random_seed: int = 42):
        """
        :param model: an initiated SKlearn model from any type
        :param scoring: a scoring string (see Sklearn docs). if not specified we use model's default
        :param prior_strategy: strategy to evaluate empty set of features (see Sklearn DummyClassifier, DummyRegressor)
        :param cv: number of cross validations to apply per evaluation
        :param random_seed: random seed for DummyClassifier
        """
        self._model = model
        self._prior_model = self._init_prior_model(prior_strategy, random_seed)
        self._cv = cv
        self._scorer = check_scoring(estimator=self._model, scoring=scoring)

    def _init_prior_model(self, prior_strategy: Optional[str], random_seed: int):
        """Init model to evaluate the empty features set performance (i.e predicting using label prior)"""

        if is_classifier(self._model):
            prior_strategy = prior_strategy if prior_strategy else "stratified"
            return DummyClassifier(strategy=prior_strategy, random_state=random_seed)
        else:
            prior_strategy = prior_strategy if prior_strategy else "mean"
            return DummyRegressor(strategy=prior_strategy)

    def __call__(self,
                 x: MultiVariateArray,
                 y: UniVariateArray,
                 x_test: Optional[MultiVariateArray] = None,
                 y_test: Optional[UniVariateArray] = None) -> float:
        """
        :param x: training x data
        :param y: training label data
        :param x_test: test x data (if None will apply cross validation on x, y)
        :param y_test: test label data (if None will apply cross validation on x, y)
        :return: the prediction score of the features in x on y
        """
        model = self._model if len(x) else self._prior_model
        x = x if not is_empty(x) else np.zeros(shape=(len(y), 1))

        if x_test is not None:
            self._model.fit(x, y)
            x_test = x_test if not is_empty(x_test) else np.zeros(shape=(len(y), 1))
            return self._scorer(self._model, x_test, y_test)
        else:
            return cross_val_score(model, x, y, cv=self._cv, scoring=self._scorer).mean()


class LgbEvaluator:

    DEAFULT_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'verbose': -1
    }

    def __init__(self, param_dict: dict = DEAFULT_PARAMS):
        self._param_dict = param_dict

    def train_lgb(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=5)
        lgb_train = lgb.Dataset(x_train, y_train, feature_name=list(x.columns))
        lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
        gbm = lgb.train(self._param_dict,
                        lgb_train,
                        valid_sets=lgb_eval,
                        verbose_eval=False)
        return gbm

    def __call__(self,
                 x,
                 y,
                 x_test,
                 y_test) -> float:
        gbm = self.train_lgb(x, y)
        y_test = y_test.astype(int)
        y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
        auc = roc_auc_score(y_test, y_pred)
        return auc

# endregions

# region tracker


class ContributionTracker:

    def __init__(self, n_features: int, track_all: bool = False):
        """
        :param n_features: number of features to track contributions for
        :param track_all: if true, saves all observed contributions and not only max per feature
        """
        self._n_features = n_features
        self.track_all = track_all

        self.max_contributions = [0.0]*self._n_features
        self.sum_contributions = [0.0]*self._n_features
        self.n_contributions = [0.0]*self._n_features
        self.argmax_contexts = [set() for _ in range(self._n_features)]

        self.all_contributions = [[] for _ in range(self._n_features)]
        self.all_contexts = [[] for _ in range(self._n_features)]

    def update_value(self, feature_idx: int, contribution: float, context: Set[str], noise_tolerance: float = 0.0):
        if contribution > self.max_contributions[feature_idx] + noise_tolerance:
            self.max_contributions[feature_idx] = contribution
            self.argmax_contexts[feature_idx] = context

        self.n_contributions[feature_idx] += 1
        self.sum_contributions[feature_idx] += contribution

        if self.track_all:
            self.all_contributions[feature_idx].append(contribution)
            self.all_contexts[feature_idx].append(context)

    def update_tracker(self, tracker: 'ContributionTracker'):
        if self.track_all and tracker.track_all:
            for feature_idx, (f_conts, f_contexts) in enumerate(zip(tracker.all_contributions, tracker.all_contexts)):
                for cont, context in zip(f_conts, f_contexts):
                    self.update_value(feature_idx, cont, context)
        else:
            for feature_idx, (cont, context) in enumerate(zip(tracker.max_contributions, tracker.argmax_contexts)):
                self.update_value(feature_idx, cont, context)

    def save_to_file(self, feature_names: List[str], file_path: str):
        state = {}

        state["max_contributions"] = self.max_contributions
        state["sum_contributions"] = self.sum_contributions
        state["n_contributions"] = self.n_contributions
        state["argmax_contexts"] = [list(c) for c in self.argmax_contexts]
        state["feature_names"] = feature_names

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @staticmethod
    def load_from_file(file_path: str, feature_names: List[str]) -> 'ContributionTracker':
        with open(file_path) as f:
            state = json.load(f)

        assert feature_names == state["feature_names"]
        tracker = ContributionTracker(n_features=len(feature_names), track_all=False)
        tracker.max_contributions = state["max_contributions"]
        tracker.sum_contributions = state["sum_contributions"]
        tracker.n_contributions = state["n_contributions"]
        tracker.argmax_contexts = state["argmax_contexts"]
        return tracker

    @property
    def avg_contributions(self):
        return [s/max(n, 1) for s, n in zip(self.sum_contributions, self.n_contributions)]

# endregion

# region mci values


class MciValues:

    """contain MCI values and project relevant plots from them"""

    def __init__(self,
                 values: Sequence[float],
                 feature_names: Sequence[str],
                 contexts: Sequence[Tuple[str, ...]],
                 additional_values: Optional[Sequence[Sequence[float]]] = None,
                 additional_contexts: Optional[Sequence[Sequence[Tuple[str, ...]]]] = None,
                 shapley_values: Optional[Sequence[float]] = None):
        """
        :param values: array of MCI values for each feature
        :param feature_names: array of features names (corresponds to the values)
        :param contexts: array of argmax contribution contexts for each feature (corresponds to the values)
        :param additional_values: placeholder for additional MCI values per feature (for non optimal values)
        :param additional_contexts: placeholder for additional MCI contexts per feature (for non optimal values)
        :param shapley_values: shapley values for comparison (optional)
        """
        self.values = values
        self.feature_names = feature_names
        self.contexts = contexts
        self.additional_values = additional_values
        self.additional_contexts = additional_contexts
        self.shapley_values = shapley_values

    @classmethod
    def create_from_tracker(cls, tracker: ContributionTracker, feature_names: Sequence[str]):
        return cls(values=tracker.max_contributions,
                   feature_names=feature_names,
                   contexts=tracker.argmax_contexts,
                   additional_values=tracker.all_contributions,
                   additional_contexts=tracker.all_contexts,
                   shapley_values=tracker.avg_contributions)
    def plot_values(self, plot_contexts: bool = False, score_name="MCI"):
        """Simple bar plot for MCI values per feature name"""
        score_features = sorted([(score, feature, context) for score, feature, context
                                 in zip(self.values, self.feature_names, self.contexts)],
                                key=lambda x: x[0])[-40:]

        if plot_contexts:
            features = [f"{f} ({', '.join(context)})" for score, f, context in score_features]
        else:
            features = [f for score, f, context in score_features]
        plt.barh(y=features, width=[score for score, f, context in score_features])
        plt.title(f"{score_name} feature importance")
        plt.xlabel(f"{score_name} value")
        plt.ylabel("Feature name")
        # plt.show()
        plt.gcf().savefig("plots/"+run_type+".mci_values.png", dpi=600,bbox_inches='tight')
        replot()

    def plot_shapley_values(self):
        score_features = sorted([(score, feature) for score, feature
                                 in zip(self.shapley_values, self.feature_names)],
                                key=lambda x: x[0])[-40:]
        features = [f for score, f in score_features]
        plt.barh(y=features, width=[score for score, f in score_features])
        plt.title(f"Shapley feature importance")
        plt.xlabel(f"Shapley value")
        plt.ylabel("Feature name")
        # plt.show()
        plt.gcf().savefig("plots/"+run_type+".shapley_values.png", dpi=600,bbox_inches='tight')
        replot()


    def results_dict(self) -> dict:
        results = {
            "feature_names": self.feature_names,
            "mci_values": self.values,
            "contexts": self.contexts,
            "shapley_values": self.shapley_values
        }
        return results

# endregion

# region estimators


class BaseEstimator(ABC):

    def __init__(self,
                 evaluator: EvaluationFunction,
                 n_processes: int = 5,
                 chunk_size: int = 20,
                 max_context_size: int = 100000,
                 noise_confidence: float = 0.05,
                 noise_factor: float = 0.1,
                 track_all: bool = False):
        """
        :param evaluator: features subsets evaluation function
        :param n_processes: number of process to use
        :param chunk_size: max number of subsets to evaluate at each process at a time
        :param max_context_size: max feature subset size to evaluate as feature context
        :param noise_confidence: PAC learning error bound confidence (usually noted as delta for PAC)
        :param noise_factor: a scalar to multiple by the PAC learning error bound
        :param track_all: a bool indicates whether to save all observed contributions and not just max
        """

        self._evaluator = evaluator
        self._n_processes = n_processes
        self._chunk_size = chunk_size
        self._max_context_size = max_context_size
        self._noise_factor = noise_factor
        self._noise_confidence = noise_confidence
        self._track_all = track_all

    @abstractmethod
    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[Sequence[str]] = None) -> MciValues:

        raise NotImplementedError()

    def _multiprocess_eval_subsets(self,
                                   subsets: List[Iterable[str]],
                                   x: DataFrame,
                                   y: UniVariateArray,
                                   x_test: Optional[DataFrame],
                                   y_test: Optional[UniVariateArray] = None) -> Dict[Tuple[str, ...], float]:
        subsets = list(set(context_to_key(c) for c in subsets))  # remove duplications

        evaluations: Dict[Tuple[str, ...], float] = {}
        pbar = tqdm(total=len(subsets))
        for eval_results in multi_process_lst(lst=subsets, apply_on_chunk=self._evaluate_subsets_chunk,
                                              chunk_size=self._chunk_size, n_processes=self._n_processes,
                                              args=(x, y, x_test, y_test)):
            evaluations.update(eval_results)
            pbar.update(len(eval_results))
        return evaluations

    def _evaluate_subsets_chunk(self,
                                subsets: List[Iterable[str]],
                                x: DataFrame,
                                y: UniVariateArray,
                                x_test: Optional[DataFrame],
                                y_test: Optional[UniVariateArray]) -> Dict[Tuple[str], float]:
        evaluations: Dict[Tuple[str, ...], float] = {}

        # print("hi!!!")
        for s in subsets:
            evaluations[context_to_key(s)] = self._evaluator(x[list(s)], y, x_test[list(s)] if x_test is not None
                    else None, y_test)
        return evaluations


class PermutationSampling(BaseEstimator):

    def __init__(self,
                 evaluator: EvaluationFunction,
                 n_permutations: int,
                 out_dir: str,
                 n_processes: int = 5,
                 chunk_size: int = 2**12,
                 max_context_size: int = 100000,
                 noise_confidence: float = 0.05,
                 noise_factor: float = 0.1,
                 track_all: bool = False,
                 permutations_batch_size: int = 200):

        super(PermutationSampling, self).__init__(evaluator=evaluator,
                                                  n_processes=n_processes,
                                                  chunk_size=chunk_size,
                                                  max_context_size=max_context_size,
                                                  noise_confidence=noise_confidence,
                                                  noise_factor=noise_factor,
                                                  track_all=track_all)
        self._n_permutations = n_permutations
        self._out_dir = out_dir
        self._n_permutations_done = 0
        self._permutations_batch_size = permutations_batch_size

    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[Sequence[str]] = None) -> MciValues:
        if not isinstance(x, DataFrame):
            assert x is not None, "feature names must be provided if x is not a dataframe"
            x = DataFrame(x, columns=feature_names)
            if x_test is not None and not isinstance(x_test, DataFrame):
                x_test = DataFrame(x_test, columns=feature_names)

        feature_names = list(x.columns)
        if os.path.isdir(self._out_dir) and len(os.listdir(self._out_dir)) > 0:
            files = [int(f.replace(".json", "")) for f in os.listdir(self._out_dir)]
            self._n_permutations_done = sorted(files)[-1]
            most_updated_file = os.path.join(self._out_dir, f"{self._n_permutations_done}.json")
            print(f"loading results checkpoint from {most_updated_file}")
            tracker = ContributionTracker.load_from_file(most_updated_file, feature_names)
        else:
            if not os.path.isdir(self._out_dir):
                os.mkdir(self._out_dir)
            tracker = ContributionTracker(n_features=len(feature_names), track_all=self._track_all)

        while self._n_permutations > self._n_permutations_done:
            np.random.seed(self._n_permutations_done)
            perm_sample_size = min(self._permutations_batch_size, self._n_permutations - self._n_permutations_done)
            permutations_sample = [list(np.random.permutation(feature_names)) for _ in range(perm_sample_size)]
            suffixes = [p[:i] for p in permutations_sample for i in range(len(p)+1)]
            evaluations = self._multiprocess_eval_subsets(suffixes, x, y, x_test, y_test)

            for p in tqdm(permutations_sample):
                for i in range(len(p)):
                    suffix = p[:i]
                    suffix_with_f = p[:i+1]
                    contribution = evaluations[context_to_key(suffix_with_f)] - evaluations[context_to_key(suffix)]
                    tracker.update_value(feature_idx=feature_names.index(p[i]),
                                         contribution=contribution,
                                         context=set(suffix))
            self._n_permutations_done += perm_sample_size
            out_path = os.path.join(self._out_dir, f"{self._n_permutations_done}.json")
            print(f"saving results for {self._n_permutations_done} permutations into {out_path}")
            tracker.save_to_file(feature_names, out_path)
        return MciValues.create_from_tracker(tracker, feature_names)

# endregion


def run_poc1(x_train, y_train, x_test, y_test):

    evaluator = LgbEvaluator()
    estimator = PermutationSampling(evaluator=evaluator, out_dir="poc1_results_checkpoints",  n_processes=1,
                                    noise_factor=0.01, chunk_size=1000, n_permutations=10,
                                    permutations_batch_size=5)

    result = estimator.mci_values(x=x_train, y=y_train, x_test=x_test, y_test=y_test)
    result.plot_values()
    result.plot_shapley_values()


def run_poc2(x_train, y_train, x_test, y_test):

    evaluator = LgbEvaluator()
    estimator = PermutationSampling(evaluator=evaluator, out_dir="poc2_results_checkpoints",  n_processes=1,
                                    noise_factor=0.01, chunk_size=5, n_permutations=40,
                                    permutations_batch_size=20)

    result = estimator.mci_values(x=x_train, y=y_train, x_test=x_test, y_test=y_test)
    result.plot_values()
    result.plot_shapley_values()


def long_run(x_train, y_train, x_test, y_test):

    evaluator = LgbEvaluator()
    estimator = PermutationSampling(evaluator=evaluator, out_dir="long_run_results_checkpoints",  n_processes=1,
                                    noise_factor=0.01, chunk_size=1000, n_permutations=2**14,
                                    permutations_batch_size=100)

    result = estimator.mci_values(x=x_train, y=y_train, x_test=x_test, y_test=y_test)
    result.plot_values()
    result.plot_shapley_values()

def histogram_intersect(a,b):
    val = np.minimum(a, b).sum().round(decimals=1)
    return val
if __name__ == '__main__':
    
    cols_to_filter = ["kupa_", "diag_", "UnitID_", "Client_", "birth_country_",
                      "pat_nation_"] #category_candida
    #13 december
    # cols_to_filter += ["Basophils_No_", "Basophils_", "Eosinophils__", "Eosinophils_No_", "Lymphocytes__", "Lymphocytes_No", "Mono_", "Monocytes_No_", "NRBC_100_WBC"]
    
    correlated_features = {'Diabetes_Drugs_Flg': 'Insulin_Drugs_Flg',
                           'category_gram_negative': 'category_gram_positive',
                           'WBC': ('Basophils_No_', 'Basophils_', 'Eosinophils__', 'Eosinophils_No_',
                                             'Lymphocytes_No','Lymphocytes__', 'Mono_', 'Monocytes_No_',
                                              'Neutrophils__', 'Neutrophils_No_'),
                           }


    train_path = r'P:\Fast_microbial_identification\model_info_november\data\real_train.csv'
    test_path = r'P:\Fast_microbial_identification\model_info_november\data\real_test.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    cols = [c for c in train.columns if not any(s in c for s in cols_to_filter)]


    y_train = train["Death"]
    y_test = test["Death"]

    # x_train = train[cols].drop(columns=["Death", "Composite"]).copy().reset_index()
    # x_test = test[cols].drop(columns=["Death", "Composite"]).copy().reset_index()

    with open(r"P:\Fast_microbial_identification\feature_lists\amnon\20_features.txt") as file:
        cols = [line.strip() for line in file]

    x_train = train[cols].copy().reset_index(drop=True)
    x_test = test[cols].copy().reset_index(drop=True)


    # corrs = (x_train.corr(method='pearson'))
 
    # corrs.to_csv('corrs_all.csv')
    # print(corrs)
    if run_type == "poc1":
        run_poc1(x_train, y_train, x_test, y_test)
    if run_type == "poc2":
        run_poc2(x_train, y_train, x_test, y_test)
    if run_type == "long":
        long_run(x_train, y_train, x_test, y_test)


    model = LgbEvaluator().train_lgb(x_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, show=False)

    plt.savefig("plots/"+run_type+".shap_summary.png", bbox_inches='tight', dpi=600)
    replot()


