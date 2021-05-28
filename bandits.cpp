#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cstdint>

using namespace std;

/***
 * DomainIds:
 * 0 - Mem
 * 1 - Cmp
 * 2 - 
 * 3 - 
 ***/
#define NUM_DOMAINS 4
#define MAX_WEIGHT 10

// C API
extern "C"
{
    /* Struct containing data for a mutation in a round of fuzzing */
    typedef struct mutation
    {
        char **argv;                      /* argv for the input. */
        int input_len;                    /* length of the buffer. */
        uint8_t *input_buffer;            /* buffer containing the input contents. */
        int fault_bit;                    /* whether or not this mutation crashed. */
        uint8_t fault_type;               /* the type of fault. See enum above. */
        uint32_t dsf_scores[NUM_DOMAINS]; /* domain specific scores. */
        struct mutation *next;            /* next mutation in the linked list. */
    } mutation;

    mutation *sample_mutation(mutation *mutations, mutation *sentinel);
}

// Function declarations
int get_advice(std::vector<std::vector<double>> &advice, mutation *mutations, mutation *sentinel);
int set_min_prob(const double p_min, std::vector<double> &prob);
int sample(const std::vector<double> &prob);
mutation *select_mutation(int index, mutation *mutations, mutation *sentinel);
int rescale_weights();

// Comparator used to iterate through probabilities by order of magnitude
struct IndexComparator
{
    explicit IndexComparator(const std::vector<double> &values)
        : values(&values) {}

    bool operator()(size_t a, size_t b)
    {
        return (*values)[a] < (*values)[b];
    }

private:
    const std::vector<double> *values;
};

// Weights maintained by this implementation. 
// - EXPERT_WEIGHTS are updated using the Exp4.P update
// - (TODO) AGGRESSIVE_WEIGHTS are updated using standard multiplicative weight 
//   update
std::vector<double> expert_weights(NUM_DOMAINS, 1.0);
// std::vector<double> aggressive_weights(NUM_DOMAINS, 1.0);
int time_step = 0;
int time_horizon = 1000;

// Logging
static ofstream log_fd;
static std::string log_fn = "afl_results/bandits.log";

/* ------------------------------------------------------------------------- */

/***
 * BANDITS VERSION 2 IMPLEMENTATION
 * --------------------------------
 * Actions - Mutations from x_t current bit string
 * Experts - Domain specific feedback sources
 * Context - If f_i is the i-th source of feedback, then it recommends the 
 ***/

/**
 * Implements one round of the contextual bandits algorithm. 
 */
mutation *sample_mutation(mutation *mutations, mutation *sentinel)
{
    // Open logging
    log_fd.open(log_fn, ios_base::app);

    // Create the advice vector
    std::vector<std::vector<double>> advice(NUM_DOMAINS,
                                            std::vector<double>(0));
    get_advice(advice, mutations, sentinel);

    // Compute the unmixed probabilities
    std::vector<double> prob(0);
    double total_prob = 0;
    int mut_size = 0;
    for (mutation *curr = mutations; curr != sentinel; curr = curr->next, mut_size++)
    {
        double mutation_prob = 0;
        double total_expert_weight = 0;

        // Compute the mutation probability
        for (int i = 0; i < NUM_DOMAINS; i++)
        {
            mutation_prob += expert_weights[i] * advice[i][mut_size];
            total_expert_weight += expert_weights[i];
        }

        // Normalize and set the probability
        mutation_prob = mutation_prob / total_expert_weight;
        prob.emplace_back(mutation_prob);
    }

    // Use the doubling trick to adaptively set the right hyper-parameters. We 
    // compute the parameters after the probability so that we'll have the size 
    // of MUTATIONS.
    // 
    // NOTE: p_min is set to the minimum between the choice to guarantee best 
    // theoretical bounds, and 1/NUM_MUT.
    if (time_step == time_horizon / 2)
    {
        time_horizon *= 2;
    }
    const double delta = 10.0;
    const double p_min = min(sqrt(log(NUM_DOMAINS) / (mut_size * time_horizon * 1.0)), 1 / (1.0 * mut_size));
    const double gamma = sqrt(log(NUM_DOMAINS / delta) / (mut_size * time_horizon * 1.0));
    log_fd << "[ITERATION " 
           << time_step 
           << "]: gamma = "
           << gamma
           << ", p_min = " 
           << p_min 
           << ", uniform = " 
           << (1 / (1.0 * mut_size)) 
           << "\n";

    // Set the minimum probability
    set_min_prob(p_min, prob);

    // Sample action
    int sampled_input = sample(prob);

    // Compute the reward estimator
    std::vector<double> reward_est(0);
    int j = 0;
    for (mutation *curr = mutations; curr != sentinel; curr = curr->next, j++)
    {
        reward_est.emplace_back(curr->fault_bit / prob[j]);
    }

    // Log the weights before...
    log_fd << "[ITERATION " << time_step << "]: weights = {";
    std::ostream_iterator<double> wt_iter_tmp(log_fd, ", ");
    std::copy(expert_weights.begin(), expert_weights.end(), wt_iter_tmp);
    log_fd << "}\n";

    // Compute the exponential update
    for (int i = 0; i < NUM_DOMAINS; i++)
    {
        double est_mean = 0;
        double est_variance = 0;

        // Compute the estimator's mean and variance
        for (int j = 0; j < prob.size(); j++)
        {
            est_mean += advice[i][j] * reward_est[j];
            est_variance += advice[i][j] / prob[j];
        }

        // Perform the exponential update
        expert_weights[i] = expert_weights[i] * exp((p_min / 2) * (est_mean + est_variance * gamma));

        log_fd << "[ITERATION " << time_step << "]: est_mean = "
               << est_mean
               << ", est_variance = "
               << est_variance
               << "\n";
        log_fd << "[ITERATION " << time_step << "]: updating expert "
               << i
               << "'s weight = "
               << exp((p_min / 2) * (est_mean + est_variance * gamma))
               << "\n";
    }

    // If the weights are too large, rescale w.r.t. the min-weight
    if (*max_element(expert_weights.begin(), expert_weights.end()) > MAX_WEIGHT) {
        log_fd << "[ITERATION " << time_step << "]: rescaling weights!";
        rescale_weights();
    }

    // Log the current weights
    log_fd << "[ITERATION " << time_step << "]: weights = {";
    std::ostream_iterator<double> wt_iter(log_fd, ", ");
    std::copy(expert_weights.begin(), expert_weights.end(), wt_iter);
    log_fd << "}\n";

    // Close logging
    log_fd.close();

    // Increment iteration count
    time_step++;

    // Output the sampled mutation
    return select_mutation(sampled_input, mutations, sentinel);
}

/**
 * Populates ADVICE[NUM_DOMAINS][num_mutations] such that ADVICE[i][j] contains
 * the i-th domain value for mutation m_j(x_t). 
 */
int get_advice(std::vector<std::vector<double>> &advice, mutation *mutations, mutation *sentinel)
{
    // Maintain a vector of context normalization values.
    std::vector<double> totals(NUM_DOMAINS);

    // Compute the unnormalized context vectors and context totals
    for (mutation *curr = mutations; curr != sentinel; curr = curr->next)
    {
        for (int i = 0; i < NUM_DOMAINS; i++)
        {
            advice[i].emplace_back(curr->dsf_scores[i] * 1.0);
            totals[i] += curr->dsf_scores[i];
        }
    }

    // Normalize the context vectors
    for (int i = 0; i < NUM_DOMAINS; i++)
    {
        for (int j = 0; j < advice[i].size(); j++)
        {
            advice[i][j] = advice[i][j] / totals[i];
        }
    }

    return 0;
}

/* ------------------------------------------------------------------------- */

/***
 * UTILITIES
 * ---------
 ***/

/**
 * Rescales the probabilities ensuring that every index of PROB is at least 
 * P_MIN. This implements algorithm 2 from Awerbach et. al.
 * 
 * PRECONDITION: P_MIN < 1/prob.size()
 */
int set_min_prob(const double p_min, std::vector<double> &prob)
{
    // Rescaling variables
    double delta = 0;
    double s = 1;

    // Get indices of prob sorted by magnitude of prob
    std::vector<size_t> indices(prob.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), IndexComparator(prob));

    // Iterate through the probabilities in sorted order
    for (const auto i : indices)
    {
        double p_old = prob[i];

        // Set the new probabilities
        if (p_old * (1 - delta / s) >= p_min)
        {
            prob[i] = p_old * (1 - delta / s);
        }
        else
        {
            prob[i] = p_min;
            delta += +prob[i] - p_old;
            s -= p_old;
        }
    }

    return 0;
}

/** 
 * Given the probability vector PROB, samples index i with probability prob[i]. 
 */
int sample(const std::vector<double> &prob)
{
    // Compute the cumulant
    std::vector<double> cumulant(prob.size());
    double curr_prob = 0;
    for (int i = 0; i < prob.size(); i++)
    {
        cumulant[i] = prob[i] + curr_prob;
        curr_prob += prob[i];
    }

    // Sample an index
    double sample = rand() / (1.0 * RAND_MAX);
    int i;
    for (i = 0; i < cumulant.size(); i++)
    {
        if (sample < cumulant[i])
        {
            return i;
        }
    }
    return i;
}

/**
 * Returns the mutation at INDEX in MUTATIONS.
 */
mutation* select_mutation(int index, mutation *mutations, mutation *sentinel)
{
    mutation *cur = mutations;
    for (int i = 0; cur != sentinel; cur = cur->next, i++)
    {
        if (i == index) {
            return cur;
        }
    }
    return cur;
}

/** 
 * Scales back the weights relative to the min-entry. 
 */
int rescale_weights() {
    double min_wt = *min_element(expert_weights.begin(), expert_weights.end()); 
    for (int i = 0; i < expert_weights.size(); i++) {
        expert_weights[i] = expert_weights[i] / min_wt;
    }
    return 0;
}