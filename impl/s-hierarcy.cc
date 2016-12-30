#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/fast-lstm.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;

float pdrop = 0.5;
unsigned LAYERS = 1;

//word
unsigned WORD_DIM = 32;
unsigned POS_DIM = 12;

unsigned HIDDEN_DIM = 100;
unsigned LABEL_HIDDEN_DIM = 100; 

unsigned LABEL_SIZE = 0;
unsigned VOCAB_SIZE = 0;

bool eval = false;
dynet::Dict wd;
dynet::Dict ld;
int kUNK;
int endlabel;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("layers", po::value<unsigned>()->default_value(1), "number of LSTM layers")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("label_dim", po::value<unsigned>()->default_value(10), "label dimension")
        ("bilstm_input_dim", po::value<unsigned>()->default_value(64), "bilstm input dimension")
        ("bilstm_hidden_dim", po::value<unsigned>()->default_value(64), "bilstm hidden dimension")
	("attention_hidden_dim", po::value<unsigned>()->default_value(64), "attention hidden dimension")
	("state_hidden_dim", po::value<unsigned>()->default_value(64), "state hidden dimension")
	("pdrop", po::value<float>()->default_value(0.3), "pdrop")
	("debug", "debug")
	("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}


template <class Builder>
struct RNNLanguageModel {
  LookupParameter p_w;
  LookupParameter p_t;
  LookupParameter p_l;
  LookupParameter p_p;

  Parameter p_oh2ohxMExp; 
  Parameter p_osh2ohxMExp;
  Parameter p_u;

  Parameter p_oh2lh;
  Parameter p_lhbias;

  Parameter p_lh2l;

  Builder l2rbuilder;
  Builder r2lbuilder;
  Builder orderbuilder;
  explicit RNNLanguageModel(Model& model) :
      l2rbuilder(LAYERS, BILSTM_INPUT_DIM, BILSTM_HIDDEN_DIM, &model),
      r2lbuilder(LAYERS, BILSTM_INPUT_DIM, BILSTM_HIDDEN_DIM, &model),
      orderbuilder(1, HIDDEN_DIM*4, HIDDEN_DIM, &model){

    p_w = model.add_lookup_parameters(VOCAB_SIZE, {WORD_DIM}); 
    p_t = model.add_lookup_parameters(POSTAG_SIZE, {POS_DIM});
    p_l = model.add_lookup_parameters(LABEL_SIZE, {LABEL_DIM});
    p_p = model.add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
    for (auto it : pretrained)
        p_p.initialize(it.first, it.second);
 
    p_oh2ohxMExp = model.add_parameters({HIDDEN_DIM, HIDDEN_DIM*2});
    p_osh2ohxMExp = model.add_parameters({HIDDEN_DIM, HIDDEN_DIM});
    p_u = model.add_parameters({1,HIDDEN_DIM});

    p_oh2lh = model.add_parameters({LABEL_HIDDEN_DIM,HIDDEN_DIM});
    p_lhbias = model.add_parameters({LABEL_HIDDEN_DIM});

    p_lh2l = model.add_parameters({LABEL_SIZE, LABEL_HIDDEN_DIM});
  }

  // return Expression of total loss
  Expression BuildTaggingGraph(const Instance& inst, ComputationGraph& cg, double* cor = 0, unsigned* predict = 0, unsigned* overall = 0, bool train=true) {
    const vector<unsigned>& sent = inst.words;
    const vector<unsigned>& sentPos = inst.postags;
    const vector<unsigned>& raws = inst.raws;
    unsigned slen = inst.words.size();
    l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);  // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    
    if(train){
      l2rbuilder.set_dropout(pdrop);
      r2lbuilder.set_dropout(pdrop);
    }
    else{
      l2rbuilder.disable_dropout();
      r2lbuilder.disable_dropout();
    }
    Expression i_oh2ohxMExp = parameter(cg, p_oh2ohxMExp);
    Expression oh2ohxMExp = parameter(cg, p_osh2ohxMExp);
    Expression i_u = parameter(cg, p_u);

    Expression i_oh2lh = parameter(cg, p_oh2lh);
    Expression i_lhbias = parameter(cg, p_lhbias);

    Expression i_lh2l = parameter(cg, p_lh2l);

    vector<Expression> i_word(slen);

    vector< vector<Expression> > i_char(slen);
    vector< vector<Expression> > i_charw(slen);
    vector< vector<Expression> > i_charh(slen);
    //attention pooling
    vector< vector<Expression> > i_charxMExp(slen);
    vector< vector<Expression> > i_charxExp(slen);
    vector< Expression > i_charSum(slen);
    vector< vector<Expression> > i_charPool(slen);
    vector< Expression > i_charinput(slen);
    //

    vector< Expression > i_input(slen);
    vector< Expression > i_inputw(slen);

    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);
    vector<Expression> i_oh(slen);

    vector<Expression> errs;

    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < VOCAB_SIZE);
      Expression w =lookup(cg, p_w, sent[i]);
      vector<Expression> args = {lb, w2l, w}; // learn embeddings        
      Expression t = lookup(cg, p_t, sentPos[i]);
      args.push_back(p2l);
      args.push_back(p);
      if (pretrained.count(raws[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(cg, p_p, raws[i]);
        args.push_back(t2l);
        args.push_back(t);
      }
      else{
        args.push_back(t2l);
        args.push_back(zeroes(cg,{PRETRAINED_DIM}));
      }
      input_expr.push_back(rectify(affine_transform(args)));
    }
 
    for (unsigned i = 0; i < slen; ++i)
      fwds[i] = l2rbuilder.add_input(input_expr[i]);
    for (unsigned i = 0; i < slen; ++i)
      revs[slen - i - 1] = r2lbuilder.add_input(input_expr[slen - i - 1]);   
    for (unsigned i = 0; i < slen; ++i)
        i_oh[i] = concatenate({fwds[i], revs[i]});

    for (unsigned i = 0; i < slen; ++i) {
   //   cerr<<i<<endl;
      orderbuilder.new_graph(cg);
      vector<Expression> l2r_c = l2rbuilder.c[i];
      vector<Expression> r2l_c = r2lbuilder.c[i];
      vector<Expression> order_init;
      for(unsigned j = 0; j < LAYERS; j ++){
        order_init.push_back(concatenate({l2r_c[j], r2l_c[j]}));
      }
      for(unsigned j = 0; j < LAYERS; j ++){
        order_init.push_back(zeroes(cg, {LABEL_HIDDEN_DIM}));
      }
      orderbuilder.start_new_sequence(order_init);
	

      int j = 0;
      Expression prev_h = orderbuilder.back();
      while(j < labels[i].size(()){
	  if(train){
	  	if(j == labels[i].size()) break;
	  }
	  else{
	  	if(j == 7) break;
	  }
	  vector<Expression> att(slen);
          for(unsigned t = 0; t < slen; t ++){
                att[t] = tanh(affine_transform({attbias, oh2att, i_oh[t], prevh2att, prev_h}));
          }
          Expression att_col = transpose(concatenate_cols(att));
          Expression attexp = softmax(att_col * att2attexp);

          Expression input_col = concatenate_cols(i_oh);
          Expression att_pool = input_col * attexp;

	  Expression rt = tanh(affine_transform({i_tbias, comb2t, concatenate({i_oh[i], i_att_oh, prev_h})}));
	  Expression le = affine_transform({i_lbias, rt2l, rt});
	    
          vector<float> dist = as_vector(cg.incremental_forward(le));
	  double best = dist[0];
	  unsigned bestk = 0;
          for (unsigned k = 1; k < dist.size(); ++k) {
            if(dist[k] > best) {best = dist[k]; bestk = k; }
          }
          if (labels[i][j] == bestk) cor += 1;
	  errs.push_back(pickneglogsoftmax(le, labels[i][j]));

          if(train) bestk = labels[i][j];
	  Expression labele = lookup(cg, p_l, bestk);	  
	  prev_h = orderbuilder.add_input({labele, att_pool}); 
          j += 1;
	  if(!train && bestk == ENDLabel) break;
      } 
      predict += j;
      overall += labels[i].size();
    }
    return sum(errs);
  }
};

class Instance{
public:
	vector<unsigned> raws;
        vector<unsigned> words;
	vector<unsigned> postags;
	vector< vector<unsigned> > labels;
        Instance(){};
	~Instance(){};
	void clear(){
		raws.clear();
		words.clear();
		labels.clear();
		postags.clear();	
	}
	void show(){
		for(unsigned i = 0; i < words.size(); i ++){
			cerr<<wd.Convert(raws[i])<<" ";
		}
		cerr<<"||| ";
		for(unsigned i = 0; i < postags.size(); i ++){
                        cerr<<td.Convert(postags[i])<<" ";
                }
                cerr<<"||| ";
		for(unsigned i = 0; i < labels.size(); i ++){
			for(unsigned j = 0; j < labels[i].size(); j ++){
				cerr<<ld.Convert(labels[i][j])<<" ";
			}
			cerr<<"||| ";
		}
		cerr<<endl;
		
	};
};

void normalize_digital_lower(string& line){
  for(unsigned i = 0; i < line.size(); i ++){
    if(line[i] >= '0' && line[i] <= '9'){
      line[i] = '0';
    }
    else if(line[i] >= 'A' && line[i] <= 'Z'){
      line[i] = line[i] - 'A' + 'a';
    }
  }
}

void ReadSentencePair(const string& line, Instance &inst){
  istringstream in(line);
  string word;
  string sep = "|||";
  int f = 0;

  while(in) {
    in >> word;
    if (word == sep) { f+=1; continue; }
    
    if(f == 0) {
      inst.raws.push_back(wd.convert(word));
    }

    else if(f == 1) {
      inst.postags.push_back(td.convert(word));
    }

    else {
      if(f == 2) inst.labels.resize(inst.raws.size());
      inst.labels[f-2].push_back(ld.convert(word));
    }
  }
}

int main(int argc, char** argv) {
  DynetParams dynet_params = extract_dynet_params(argc, argv);
  dynet_params.random_seed = 1989121013;
  dynet::initialize(dynet_params);
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;

  kUNK = wd.convert("*UNK*");
  if (conf.count("words")) {
    pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
    cerr << "Loading from " << conf["words"].as<string>() << " with" << PRETRAINED_DIM << " dimensions\n";
    ifstream in(conf["words"].as<string>().c_str());
    string line;
    getline(in, line);
    vector<float> v(PRETRAINED_DIM, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
      unsigned id = corpus.get_or_add_word(word);
      pretrained[id] = v;
    }
  }

  vector<Instance> training,dev,test;
  cerr << "Reading training data from " << conf["training_data"].as<string>() << "...\n";
  {
    unsigned ttoks = 0;
    unsigned unk = 0;
    ifstream in(conf["training_data"].as<string>().c_str());
    assert(in);
    string line;
    while(getline(in, line)) {
      Instance inst; 
      ReadSentencePair(line, inst);
      training.push_back(inst);
      ttoks += x.size();
    }
    for (auto& sent : training){
        const vector<unsigned>& raws = sent.raws;
	for (unsigned i = 0; i < raws.size(); ++i){
          training_vocab.insert(raws[i]); counts[raws[i]]++;
	}
      }
    }
    for (auto wc : counts) if (wc.second == 1) singletons.insert(wc.first); 
    
    for (auto& sent : training){
	const vector<unsigned>& raws = sent.raws;
        vector<unsigned>& words = sent.words;
	words.resize(raws.size());
	for (unsigned i = 0; i < raws.size(); ++i){
	  if(singetons.find(words[i]) && dynet::rand01() < unk_prob) {words[i] = kUNK; unk += 1;}
	  else {words[i] = raws[i];}
	}
    }
    cerr << training.size() << " lines, " << ttoks << " tokens, " << unk << " UNK "
         << "OOV ratio: " << float(unk)/ttoks<<"\n";
  }

  cerr<< "Rading dev data from " << conf["dev_data"].as<string>() << "...\n";
  {
    unsigned ttoks = 0;
    unsigned unk = 0;
    ifstream in(conf["dev_data"].as<string>().c_str());
    assert(in);
    string line;
    while(getline(in, line)) {
      Instance inst; 
      ReadSentencePair(line, inst);
      dev.push_back(inst);
      ttoks += x.size();
    }
    for (auto& sent : dev){
	const vector<unsigned>& raws = sent.raws;
        vector<unsigned>& words = sent.words;
	words.resize(raws.size());
        for (unsigned i = 0; i < words.size(); ++i){
          if(!training_vocab.find(words[i])) {words[i] = kUNK; unk += 1;}
          else {words[i] = raws[i];}
	}
    }
    cerr << dev.size() << " lines, " << ttoks << " tokens, " << unk << " UNK "
         << "OOV ratio: " << float(unk)/ttoks<<"\n";
  }


  cerr<< "Rading test data from " << conf["test_data"].as<string>() << "...\n";
  { 
    unsigned ttoks = 0;
    unsigned unk = 0;
    ifstream in(conf["test_data"].as<string>().c_str());
    assert(in);
    string line;
    while(getline(in, line)) {
      Instance inst; 
      ReadSentencePair(line, inst);
      dev.push_back(inst);
      ttoks += x.size();
    }
    for (auto& sent : test){
        const vector<unsigned>& raws = sent.raws;
        vector<unsigned>& words = sent.words;
        words.resize(raws.size());
        for (unsigned i = 0; i < words.size(); ++i){
          if(!training_vocab.find(words[i])) {words[i] = kUNK; unk += 1;}
          else {words[i] = raws[i];}
        }
    }
    cerr << test.size() << " lines, " << ttoks << " tokens, " << unk << " UNK "
         << "OOV ratio: " << float(unk)/ttoks<<"\n";
  }


  VOCAB_SIZE = wd.size();
  POSTAG_SIZE = td.size();
  LABEL_SIZE = ld.size();

  ostringstream os;
  os << "ordered"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = -9e+99;

  Model model;
  
  bool use_momentum = true;
  Trainer* sgd = nullptr;
  unsigned method = conf["train_methods"].as<unsigned>();
    if(method == 0)
  	sgd = new SimpleSGDTrainer(&model,0.1, 0.1);
    else if(method == 1)
	sgd = new MomentumSGDTrainer(&model,0.01, 0.9, 0.1);
    else if(method == 2){
	sgd = new AdagradTrainer(&model);
	sgd->clipping_enabled = false;	
    }
    else if(method == 3){
	sgd = new AdamTrainer(&model);
  	sgd->clipping_enabled = false;
    } 

  RNNLanguageModel<FastLSTMBuilder> lm(model);
  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (conf.count("model")) {
    string fname = conf["model"].as<string>();
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }


  if(conf.count("train")){
  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned tpredict = 0;
    unsigned toverall = 0;
    double correct = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }
      ComputationGraph cg;
      auto& sent = training[order[si]];
      ++si;
      Expression nll = lm.BuildTaggingGraph(sent, cg, &correct, &tpredict, &toverall, true);
      loss += as_scalar(cg.forward(nll));
      cg.backward(nll);
      sgd->update(1.0);
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / toverall) << " ppl=" << exp(loss / toverall) << " (acc=" << (correct / toverall) << ") ";

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dpredict = 0;
      unsigned doverall = 0;
      double dcorr = 0;
      eval = true;
      //lm.p_th2t->scale_parameters(pdrop);
      for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildTaggingGraph(sent.words, sent.chars, sent.labels, cg, &dcorr, &dpredict, &doverall, false);
        dloss += as_scalar(cg.forward());
      }
      //lm.p_th2t->scale_parameters(1/pdrop);
      eval = false;
      double P = (dcorr / dpredict);
      double R = (dcorr / doverall);
      double F = 2*P*R / (P+R);
      if (F > best) {
        best = F;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] P = " << P << " R = " << R << " F=" << F << ' ';
    }
  }
  }
  else{
  }
  delete sgd;
}

