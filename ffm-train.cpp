#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

#include "ffm.h"

using namespace std;
using namespace ffm;

string train_help()
{
    return string(
"usage: ffm-train [options] training_set_file [model_file]\n"
"\n"
"options:\n"
"-l <lambda>: set regularization parameter (default 0.00002)\n"
"-k <factor>: set number of latent factors (default 4)\n"
"-t <iteration>: set number of iterations (default 15)\n"
"-r <eta>: set learning rate (default 0.2)\n"
"-s <nr_threads>: set number of threads (default 1)\n"
"-p <path>: set path to the validation set\n"
"-v <fold>: set the number of folds for cross-validation\n"
"--quiet: quiet model (no output)\n"
"--no-norm: disable instance-wise normalization\n"
"--no-rand: disable random update\n"
"--on-disk: perform on-disk training (a temporary file <training_set_file>.bin will be generated)\n"
"--auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)\n"
"--block-structure <path>: set path to block structure\n"
"--negative-probabilities <path>: set path to feature negative probabilities (also enable negative sampling)"
"--negative-samples: set number of negative samples (default 5)"
"--negative-position: sets the position of feature to be negatively sampled");
}

struct Option
{
    Option() : param(ffm_get_default_param()), nr_folds(1), do_cv(false), on_disk(false) {}
    string tr_path, va_path, model_path, bs_path, neg_path;
    ffm_parameter param;
    ffm_int nr_folds;
    bool do_cv, on_disk, do_bs, do_neg;
};

string basename(string path)
{
    const char *ptr = strrchr(&*path.begin(), '/');
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option opt;

    ffm_int i = 1;
    for(; i < argc; i++)
    {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            opt.param.nr_iters = atoi(args[i].c_str());
            if(opt.param.nr_iters <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        }
        else if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            opt.param.k = atoi(args[i].c_str());
            if(opt.param.k <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify eta after -r");
            i++;
            opt.param.eta = atof(args[i].c_str());
            if(opt.param.eta <= 0)
                throw invalid_argument("learning rate should be greater than zero");
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify lambda after -l");
            i++;
            opt.param.lambda = atof(args[i].c_str());
            if(opt.param.lambda < 0)
                throw invalid_argument("regularization cost should not be smaller than zero");
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            opt.param.nr_threads = atoi(args[i].c_str());
            if(opt.param.nr_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of folds after -v");
            i++;
            opt.nr_folds = atoi(args[i].c_str());
            if(opt.nr_folds <= 1)
                throw invalid_argument("number of folds should be greater than one");
            opt.do_cv = true;
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;
            opt.va_path = args[i];
        }
        else if(args[i].compare("--no-norm") == 0)
        {
            opt.param.normalization = false;
        }
        else if(args[i].compare("--quiet") == 0)
        {
            opt.param.quiet = true;
        }
        else if(args[i].compare("--no-rand") == 0)
        {
            opt.param.random = false;
        }
        else if(args[i].compare("--on-disk") == 0)
        {
            opt.on_disk = true;
        }
        else if(args[i].compare("--auto-stop") == 0)
        {
            opt.param.auto_stop = true;
        }
        else if(args[i].compare("--block-structure") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after --block-structure");
            i++;
            opt.bs_path = args[i].c_str();
            opt.do_bs = true;
        }
        else if(args[i].compare("--negative-probabilities") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after --negative-probabilities");
            i++;
            opt.neg_path = args[i].c_str();
            opt.do_neg = true;
        }
        else if(args[i].compare("--negative-samples") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of samples after --negative-samples");
            i++;
            opt.param.negative_samples = atoi(args[i].c_str());
            if(opt.param.negative_samples < 1)
              throw invalid_argument("number of negative samples should be larger than zero");
        }
        else if(args[i].compare("--negative-position") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify negative position after --negative-position");
            i++;
            opt.param.negative_position = atoi(args[i].c_str());
            if(opt.param.negative_position >= 0)
              throw invalid_argument("number of negative samples should be larger or equal to zero");
        }
        else
        {
            break;
        }
    }

    if(i != argc-2 && i != argc-1)
        throw invalid_argument("cannot parse command\n");

    opt.tr_path = args[i];
    i++;

    if(i < argc)
    {
        opt.model_path = string(args[i]);
    }
    else if(i == argc)
    {
        opt.model_path = basename(opt.tr_path) + ".model";
    }
    else
    {
        throw invalid_argument("cannot parse argument");
    }

    return opt;
}

const ffm_int num_negative_sampling_buckets = 1e8;

int train(Option opt)
{
    ffm_negative_sampling *ns=nullptr;
    if(opt.do_neg)
    {
        if(opt.param.negative_position<0)
            throw invalid_argument("no negative position has been specified");
        // XXX(AR): Destroy negative sampling
        ns = ffm_create_negative_sampling(opt.param.negative_position, opt.param.negative_samples,
                                          opt.neg_path.c_str(), num_negative_sampling_buckets);
        if(ns == nullptr)
            throw invalid_argument("cannot initialize negative sampling");
    }
    ffm_block_structure *bs=nullptr;
    if(opt.do_bs)
    {
        // XXX(AR): Destroy block structure
        bs = ffm_read_block_structure(opt.bs_path.c_str());
        if(bs == nullptr)
            throw invalid_argument("cannot load block structure");
    }
    ffm_problem *tr = ffm_read_problem(opt.tr_path.c_str());
    if(tr == nullptr)
    {
        cerr << "cannot load " << opt.tr_path << endl << flush;
        ffm_destroy_block_structure(&bs);
        return 1;
    }

    ffm_problem *va = nullptr;
    if(!opt.va_path.empty())
    {
        va = ffm_read_problem(opt.va_path.c_str());
        if(va == nullptr)
        {
            ffm_destroy_problem(&tr);
            ffm_destroy_block_structure(&bs);
            cerr << "cannot load " << opt.va_path << endl << flush;
            return 1;
        }
    }

    int status = 0;
    if(opt.do_cv)
    {
        ffm_cross_validation(tr, opt.nr_folds, bs, opt.param);
    }
    else
    {
        ffm_model *model = ffm_train_with_validation(tr, va, bs, opt.param);

        status = ffm_save_model(model, opt.model_path.c_str());

        ffm_destroy_model(&model);
    }

    ffm_destroy_block_structure(&bs);
    ffm_destroy_problem(&tr);
    ffm_destroy_problem(&va);

    return status;
}

int train_on_disk(Option opt)
{
    ffm_block_structure *bs=nullptr;
    if(opt.do_bs)
    {
      // XXX(AR): Destroy block structure
      bs = ffm_read_block_structure(opt.bs_path.c_str());
      if(bs == nullptr)
      {
          cerr << "cannot load " << opt.bs_path << endl << flush;
          return 1;
      }
    }

    if(opt.param.random)
    {
        cout << "Random update is not allowed in disk-level training. Please use `--no-rand' to disable." << endl;
        return 1;
    }

    if(opt.do_cv)
    {
        cout << "Cross-validation is not yet implemented in disk-level training." << endl;
        return 1;
    }

    string tr_bin_path = basename(opt.tr_path) + ".bin";
    string va_bin_path = opt.va_path.empty()? "" : basename(opt.va_path) + ".bin";

    ffm_read_problem_to_disk(bs, opt.tr_path.c_str(), tr_bin_path.c_str());
    if(!opt.va_path.empty())
        ffm_read_problem_to_disk(bs, opt.va_path.c_str(), va_bin_path.c_str());

    ffm_model *model = ffm_train_with_validation_on_disk(tr_bin_path.c_str(), va_bin_path.c_str(), bs, opt.param);

    ffm_int status = ffm_save_model(model, opt.model_path.c_str());
    if(status != 0)
    {
        ffm_destroy_block_structure(&bs);
        ffm_destroy_model(&model);

        return 1;
    }

    ffm_destroy_block_structure(&bs);
    ffm_destroy_model(&model);

    return 0;
}

int main(int argc, char **argv)
{
    Option opt;
    try
    {
        opt = parse_option(argc, argv);
    }
    catch(invalid_argument &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    if(opt.on_disk)
    {
        return train_on_disk(opt);
    }
    else
    {
        return train(opt);
    }
}
