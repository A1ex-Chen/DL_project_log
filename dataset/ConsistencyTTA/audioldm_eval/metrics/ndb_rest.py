import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
from matplotlib import pyplot as plt
import pickle as pkl


class NDB:

        # print('Done.')







    @staticmethod

    @staticmethod

    @staticmethod

        if not models_to_plot:
            models_to_plot = sorted(list(self.cached_results.keys()))

        # Visualize the standard errors using the train proportions and size and query sample size
        train_se = calc_se(
            self.bin_proportions,
            self.ref_sample_size,
            self.bin_proportions,
            self.cached_results[models_to_plot[0]]["N"],
        )
        plt.bar(
            np.arange(0, K) + 0.5,
            height=train_se * 2.0,
            bottom=self.bin_proportions - train_se,
            width=1.0,
            label="Train$\pm$SE",
            color="gray",
        )

        ymax = 0.0
        for i, model in enumerate(models_to_plot):
            results = self.cached_results[model]
            label = "%s (%i : %.4f)" % (model, results["NDB"], results["JS"])
            ymax = max(ymax, np.max(results["Proportions"]))
            if K <= 70:
                plt.bar(
                    np.arange(0, K) + (i + 1.0) * w,
                    results["Proportions"],
                    width=w,
                    label=label,
                )
            else:
                plt.plot(
                    np.arange(0, K) + 0.5, results["Proportions"], "--*", label=label
                )
        plt.legend(loc="best")
        plt.ylim((0.0, min(ymax, np.max(self.bin_proportions) * 4.0)))
        plt.grid(True)
        plt.title(
            "Binning Proportions Evaluation Results for {} bins (NDB : JS)".format(K)
        )
        plt.show()

    def __calculate_bin_proportions(self, samples):
        if self.bin_centers is None:
            print(
                "First run construct_bins on samples from the reference training data"
            )
        # print("as1",samples.shape[1])
        # print("as2",self.bin_centers.shape[1])
        assert samples.shape[1] == self.bin_centers.shape[1]
        n, d = samples.shape
        k = self.bin_centers.shape[0]
        D = np.zeros([n, k], dtype=samples.dtype)

        # print('Calculating bin assignments for {} samples...'.format(n))
        whitened_samples = (samples - self.training_mean) / self.training_std
        for i in range(k):
            print(".", end="", flush=True)
            D[:, i] = np.linalg.norm(
                whitened_samples[:, self.used_d_indices]
                - self.bin_centers[i, self.used_d_indices],
                ord=2,
                axis=1,
            )
        print()
        labels = np.argmin(D, axis=1)
        probs = np.zeros([k])
        label_vals, label_counts = np.unique(labels, return_counts=True)
        probs[label_vals] = label_counts / n
        return probs, labels

    def __read_from_bins_file(self, bins_file):
        if bins_file and os.path.isfile(bins_file):
            print("Loading binning results from", bins_file)
            bins_data = pkl.load(open(bins_file, "rb"))
            self.bin_proportions = bins_data["proportions"]
            self.bin_centers = bins_data["centers"]
            self.ref_sample_size = bins_data["n"]
            self.training_mean = bins_data["mean"]
            self.training_std = bins_data["std"]
            self.used_d_indices = bins_data["d_indices"]
            return True
        return False

    def __write_to_bins_file(self, bins_file):
        if bins_file:
            print("Caching binning results to", bins_file)
            bins_data = {
                "proportions": self.bin_proportions,
                "centers": self.bin_centers,
                "n": self.ref_sample_size,
                "mean": self.training_mean,
                "std": self.training_std,
                "d_indices": self.used_d_indices,
            }
            pkl.dump(bins_data, open(bins_file, "wb"))

    @staticmethod
    def two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None):
        # Per http://stattrek.com/hypothesis-test/difference-in-proportions.aspx
        # See also http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        # print("z",abs(z))
        # Allow defining a threshold in terms as Z (difference relative to the SE) rather than in p-values.
        if z_threshold is not None:
            return abs(z) > z_threshold
        p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))  # Two-tailed test
        return p_values < significance_level

    @staticmethod
    def jensen_shannon_divergence(p, q):
        """
        Calculates the symmetric Jensen–Shannon divergence between the two PDFs
        """
        m = (p + q) * 0.5
        return 0.5 * (NDB.kl_divergence(p, m) + NDB.kl_divergence(q, m))

    @staticmethod
    def kl_divergence(p, q):
        """
        The Kullback–Leibler divergence.
        Defined only if q != 0 whenever p != 0.
        """
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(q))
        assert not np.any(np.logical_and(p != 0, q == 0))

        p_pos = p > 0
        return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))


if __name__ == "__main__":
    dim = 100
    k = 100
    n_train = k * 100
    n_test = k * 10

    train_samples = np.random.uniform(size=[n_train, dim])
    ndb = NDB(training_data=train_samples, number_of_bins=k, whitening=True)

    test_samples = np.random.uniform(high=1.0, size=[n_test, dim])
    ndb.evaluate(test_samples, model_label="Test")

    test_samples = np.random.uniform(high=0.9, size=[n_test, dim])
    ndb.evaluate(test_samples, model_label="Good")

    test_samples = np.random.uniform(high=0.75, size=[n_test, dim])
    ndb.evaluate(test_samples, model_label="Bad")

    ndb.plot_results(models_to_plot=["Test", "Good", "Bad"])