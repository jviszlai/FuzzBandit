#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>

using namespace std;

/***
 * DomainIds:
 * 0 - Mem
 * 1 - Cmp
 * 2 - 
 * 3 - 
 ***/
#define NUM_DOMAINS 2

// The minimum probability to play
// TODO: make this a parameter into sample_input?
// TODO: actually set the right parameter for GAMMA
#define P_MIN 0.0001
#define GAMMA 0.0005
#define REWARD_SCALE 1000

// C API
extern "C"
{
    typedef struct {
        char *out_buf;
        int len;
        char fault;
        char **argv;
        char *op_descript;
    } mutation_buf;

    typedef struct mutation {
        int stage_id;
        int index;
        int feedback[NUM_DOMAINS];
        unsigned int hash;
        int did_crash;
        mutation_buf file;
        struct mutation *next;
    } mutation;

    int sample_input_1(mutation *mutations, unsigned int curr_input);
    int sample_input_2(mutation *mutations, unsigned int curr_input);
}

// Function declarations
int get_advice_1(mutation *mutations, std::vector<double> &advice);
int set_min_prob(const double p_min, std::vector<double> &prob);
int sample(const std::vector<double> &prob);

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

// BANDITS VERSION 1 IMPLEMENTATION constants
double domain_weights[] = {1, 10000000000000};
std::map<unsigned int, double> expert_weights_1;
std::set<unsigned int> visited_inputs;

// BANDITS VERSION 2 IMPLEMENTATION constants
std::vector<double> expert_weights_2(NUM_DOMAINS, 1.0);

// Logging
static ofstream log_file;
static std::string log_file_name = "prob_log.txt";


/* ------------------------------------------------------------------------- */


/***
 * BANDITS VERSION 1 IMPLEMENTATION
 * --------------------------------
 * Actions - Mutations from x_t current bit string
 * Experts - Set of all bit strings
 * Context - If "a" is an expert such that a = m_j(x_t) for one of the 
 *           mutations then "a" recommends the aggregate score for m_j(x_t)
 ***/

// Everytime this is called, do one iteration of multi-armed bandits
// Write file into queue location
int sample_input_1(mutation *mutations, unsigned int curr_input)
{
    // log_file.open(log_file_name, std::ios_base::app);

    // log_file << "================================\n\n\n REPORTS \n\n\n";
    // mutation *curr = mutations;
    // while (curr) {
    //     log_file << curr->feedback[1] << ", ";
    //     curr = curr->next;
    // }

    // Compute advice
    std::vector<double> advice(0);
    get_advice_1(mutations, advice);

    // Compute unmixed probabilities
    std::vector<double> prob(0);
    double total_prob = 0;
    
    mutation *curr = mutations;
    for (int i = 0; curr; curr = curr->next, i++)
    {
        if (expert_weights_1.find(curr->hash) == expert_weights_1.end())
        {
            expert_weights_1[curr->hash] = 1;
        }
        prob.emplace_back(expert_weights_1[curr->hash] * advice[i]);
        total_prob = total_prob + prob[i];
    }
    for (int i = 0; i < prob.size(); i++)
    {
        prob[i] = prob[i] / total_prob;
    }

    // Ensure that all actions are played with p_min
    set_min_prob(P_MIN, prob);

    // Sample action
    int sampled_input = sample(prob);

    // Compute reward estimator
    std::vector<double> reward_est(0);
    curr = mutations;
    for (int i = 0; curr; curr = curr->next, i++)
    {
        reward_est.emplace_back(REWARD_SCALE * curr->did_crash / prob[i]);
    }

    // Compute an exponential weight update
    curr = mutations;
    for (int i = 0; curr; curr = curr->next, i++)
    {
        expert_weights_1[curr->hash] = expert_weights_1[curr->hash] 
            * exp((P_MIN / 2) * advice[i] * (reward_est[i] + GAMMA / prob[i]));
    }

    // Add x_t to the seen set
    visited_inputs.insert(curr_input);

    // std::ostringstream oss;
    // log_file << "\n\n\n PROB \n\n\n\ ";
    // std::copy(prob.begin(), prob.end()-1000,
    //     std::ostream_iterator<double>(oss, ", "));

    // log_file << oss.str();

    // std::ostringstream oss2;
    // log_file << "\n\n\n ADVICE \n\n\n\ ";
    // std::copy(advice.begin(), advice.end()-1000,
    //     std::ostream_iterator<double>(oss2, ", "));

    // log_file << oss2.str();
    // log_file << "\n\n MUTATION \n\n";
    // log_file << mutation;

    // log_file.close();

    // Output sampled mutation
    return sampled_input;
}

// Returns the aggregate score from the domain_reports
int get_advice_1(mutation *mutations, std::vector<double> &advice)
{
    // Compute aggregate score for each action
    double total_score = 0;
    for (mutation *curr = mutations; curr; curr = curr->next)
    {
        double aggregate_score = 0;
        for (int j = 0; j < NUM_DOMAINS; j++)
        {
            aggregate_score += domain_weights[j] * curr->feedback[j];
        }
        advice.emplace_back(aggregate_score);
        total_score += aggregate_score;
    }

    // Normalize the feedback
    for (int i = 0; i < advice.size(); i++)
    {
        advice[i] = advice[i] / total_score;
    }
}


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
int sample_input_2(mutation *mutations, unsigned int curr_input)
{
    // Bandit hyperparameters (TODO actually set these)
    double p_min = 0.0001;
    double gamma = 0.0005;
    
    // Create the advice vector
    std::vector<std::vector<double>> advice(NUM_DOMAINS, 
                                            std::vector<double>(0));
    get_advice_2(mutations, advice);

    // Compute the unmixed probabilities
    std::vector<double> prob(0);
    double total_prob = 0;
    int j = 0;
    for (mutation *curr = mutations; curr; curr = curr->next, j++) 
    {
        double mutation_prob = 0;
        double total_expert_weight = 0;

        // Compute the mutation probability
        for (int i = 0; i < NUM_DOMAINS; i++) 
        {
            mutation_prob += expert_weights_2[i] * advice[i][j];
            total_expert_weight += expert_weights_2[i];
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
    for (mutation *curr = mutations; curr; curr = curr->next, j++) 
    {
        reward_est.emplace_back(curr->did_crash / prob[j]);
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
        expert_weights_2[i] = expert_weights_2[i] 
                * exp((p_min / 2) * (est_mean + est_variance * gamma));
    }

    // Output the sampled mutation
    return sampled_input;
}

/**
 * Populates ADVICE[NUM_DOMAINS][num_mutations] such that ADVICE[i][j] contains
 * the i-th domain value for mutation m_j(x_t). 
 */
int get_advice_2(mutation *mutations, 
                 std::vector<std::vector<double>> &advice) {
    // Maintain a vector of context normalization values.
    std::vector<double> totals(NUM_DOMAINS);

    // Compute the unnormalized context vectors and context totals
    for (mutation *curr = mutations; curr; curr = curr->next)
    {
        for (int i = 0; i < NUM_DOMAINS; i++) 
        {
            advice[i].emplace_back(curr->feedback[i] * 1.0);
            totals[i] += curr->feedback[i];
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
