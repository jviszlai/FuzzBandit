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

// Expert weights maintained by this implementation
std::vector<double> expert_weights(NUM_DOMAINS, 1.0);

// Logging
static ofstream log_file;
static std::string log_file_name = "bandits.log";

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
    // Bandit hyperparameters (TODO actually set these)
    double p_min = 0.0001;
    double gamma = 0.0005;

    // Create the advice vector
    std::vector<std::vector<double>> advice(NUM_DOMAINS,
                                            std::vector<double>(0));
    get_advice(advice, mutations, sentinel);

    // Compute the unmixed probabilities
    std::vector<double> prob(0);
    double total_prob = 0;
    int j = 0;
    for (mutation *curr = mutations; curr != sentinel; curr = curr->next, j++)
    {
        double mutation_prob = 0;
        double total_expert_weight = 0;

        // Compute the mutation probability
        for (int i = 0; i < NUM_DOMAINS; i++)
        {
            mutation_prob += expert_weights[i] * advice[i][j];
            total_expert_weight += expert_weights[i];
        }

        // Normalize and set the probability
        mutation_prob = mutation_prob / total_expert_weight;
        prob.emplace_back(mutation_prob);
    }

    // Set the minimum probability
    set_min_prob(p_min, prob);

    // Sample action
    int sampled_input = sample(prob);

    // Compute the reward estimator
    std::vector<double> reward_est(0);
    j = 0;
    for (mutation *curr = mutations; curr != sentinel; curr = curr->next, j++)
    {
        reward_est.emplace_back(curr->fault_bit / prob[j]);
    }

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
    }

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