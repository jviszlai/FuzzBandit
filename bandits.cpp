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
 * Feedback IDs:
 * 0 - domain specific feedback 1
 * 1 - domain specific feedback 2
 * 2 - domain specific feedback 3
 * 3 - domain specific feedback 4
 * 4 - AFL compute_score() value
 * 5 - code coverage (bitmap size)
 * 6 - execution time (in microseconds... but probably translated to seconds)
 ***/
#define NUM_DOMAINS 4
#define NUM_FEEDBACK 7
#define MAX_WEIGHT 10

// C API
extern "C"
{
    /* The queue_entry struct used by afl-fuzz. */
    struct queue_entry
    {
        uint8_t *fname;   /* File name for the test case      */
        uint32_t len;     /* Input length                     */
        uint8_t *buf;     /* Input buffer.                    */
        char **argv;      /* argv for the input.              */

        uint8_t cal_failed,    /* Calibration failed?              */
                trim_done,    /* Trimmed?                         */
                was_fuzzed,   /* Had any fuzzing done yet?        */
                passed_det,   /* Deterministic stages passed?     */
                has_new_cov,  /* Triggers new coverage?           */
                var_behavior, /* Variable behavior?               */
                favored,      /* Currently favored?               */
                fs_redundant; /* Marked as redundant in the fs?   */

        uint32_t bitmap_size, /* Number of bits set in bitmap     */
                 exec_cksum,  /* Checksum of the execution trace  */
                 dsf_cksum;   /* DSF - cksum of unbukceted trace */

        uint64_t exec_us,  /* Execution time (us)              */
                 handicap, /* Number of queue cycles behind    */
                 depth;    /* Path depth                       */

        uint8_t *trace_mini; /* Trace bytes, if kept             */
        uint32_t tc_ref;     /* Trace bytes ref count            */

        struct queue_entry *next; /* Next element, if any             */
    };

    /* Struct containing data for a mutation in a round of fuzzing */
    typedef struct mutation
    {
        int fault_bit;                  /* whether or not this mutation crashed. */
        uint8_t fault_type;             /* the type of fault. See enum above. */
        uint32_t dsf_scores[NUM_DOMAINS];   /* domain specific scores. */
        uint32_t afl_score;             /* the score used by AFL. */
        uint32_t bitmap_size;           /* measures code coverage. */
        uint64_t exec_us;               /* average exec time computed in calibration. */
        struct queue_entry *mut_q;      /* the queue entry for this mutation. */
        struct mutation *next;          /* next mutation in the linked list. */
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

// Weights maintained using the Exp4.P update
std::vector<double> expert_weights(NUM_FEEDBACK, 1.0);
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
        for (int i = 0; i < expert_weights.size(); i++)
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
    const double delta = 2.0; // this should definitely be < NUM_FEEDBACK
    const double p_min = min(sqrt(log(expert_weights.size()) / (mut_size * time_horizon * 1.0)), 1 / (1.0 * mut_size));
    const double gamma = sqrt(log(expert_weights.size() / delta) / (mut_size * time_horizon * 1.0));
    
    // log_fd << "[ITERATION " 
    //        << time_step 
    //        << "]: gamma = "
    //        << gamma
    //        << ", p_min = " 
    //        << p_min 
    //        << ", uniform = " 
    //        << (1 / (1.0 * mut_size)) 
    //        << "\n";
    // log_fd << "[ITERATION "
    //        << time_step
    //        << "]: logging gamma quantities\n"
    //        << "- log-term: "
    //        << log(expert_weights.size() / delta)
    //        << "\n- denominator: "
    //        << (mut_size * time_horizon * 1.0)
    //        << "\n- divided: "
    //        << log(expert_weights.size() / delta) / (mut_size * time_horizon * 1.0)
    //        << "\n- sqrt-rootd: "
    //        << sqrt(log(expert_weights.size() / delta) / (mut_size * time_horizon * 1.0))
    //        << "\n";

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

    // Compute the exponential update
    for (int i = 0; i < expert_weights.size(); i++)
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

    // If the weights are too large, rescale w.r.t. the min-weight
    if (*max_element(expert_weights.begin(), expert_weights.end()) > MAX_WEIGHT) {
        log_fd << "[ITERATION " << time_step << "]: rescaling weights!\n";
        rescale_weights();
    }

    // Log the current weights
    log_fd << "[ITERATION " << time_step << "]: "
           << "# mutations = {" << mut_size << "}, "
           << "weights = {";
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
 * Populates ADVICE[NUM_FEEDBACK][num_mutations] such that ADVICE[i][j] contains
 * the i-th domain value for mutation m_j(x_t). 
 */
int get_advice(std::vector<std::vector<double>> &advice, mutation *mutations, mutation *sentinel)
{
    // Maintain a vector of context normalization values.
    std::vector<double> totals(NUM_FEEDBACK);

    // Compute the unnormalized context vectors and context totals
    for (mutation *curr = mutations; curr != sentinel; curr = curr->next)
    {
        for (int i = 0; i < NUM_DOMAINS; i++)
        {
            advice[i].emplace_back(curr->dsf_scores[i] * 1.0);
            totals[i] += curr->dsf_scores[i] * 1.0;
        }

        // Compute feedback for the AFL compute_score() value
        advice[4].emplace_back(curr->afl_score * 1.0);
        totals[4] += curr->afl_score * 1.0;

        // Compute feedback for code coverage
        advice[5].emplace_back(curr->bitmap_size * 1.0);
        totals[5] += curr->bitmap_size * 1.0;

        // Compute feedback for execution time
        double exec_sec = curr->bitmap_size / (1.0 * 1e6);
        advice[6].emplace_back(exec_sec);
        totals[6] += exec_sec;
    }

    // Normalize the context vectors
    for (int i = 0; i < NUM_FEEDBACK; i++)
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